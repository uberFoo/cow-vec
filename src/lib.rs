//! CowVec: A vector of CoW elements
//! The intended purpose of this implementation is to provide a means to create
//! cheap references to the elements in another `Vec`. As long as there is no
//! mutation, access is fast, and the memory cost is low.
//!
//! If you want to mutate an element, we utilize [XuderCow] to create a new Boxed
//! element which is mutable, and does not effect the other `Vec`.
//!
//! This implementation uses unsafe code to store the elements. We allocate
//! memory by means of a `Vec`, who's len is always zero.
//!
//! I wrote this partly for fun, but mostly because I needed a collection that
//! I could implement mutable iterators on. I was also hoping that I could get
//! mutable and immutable references to elements at the same time. Only because
//! the `XuderCow` can change its referent to a `Box<T>` without invalidating the
//! references that were pointing at the `&T`. Maybe I can do something tricky
//! with lifetimes.
use core::{
    cmp::PartialEq,
    fmt::Debug,
    iter::FromIterator,
    ops::{Deref, DerefMut},
};

/// A vector with CoW elements
///
#[derive(Debug)]
pub struct CowVec<'a, T>
where
    T: Clone,
{
    storage: Vec<XuderCow<'a, T>>,
    len: usize,
}

// Public interface
impl<'a, T> CowVec<'a, T>
where
    T: Clone,
{
    /// Create a new `CowVec`
    ///
    /// Note that the only way to properly construct one of these is to pass a
    /// reference to a `Vec<T>`. We use our `FromIterator` implementation to
    /// create a new `CowVec` that contains references to the input `Vec`.
    /// Of course those references are wrapped in `XuderCow`s for trickery. Well
    /// and becuase that's the point of a CowVec...
    pub fn new(source: &'a Vec<T>) -> Self {
        source.iter().map(|a| a).collect::<CowVec<T>>()
    }

    /// Get our capacity
    ///
    /// Capacity is in number of elements we can hold.
    pub fn capacity(&self) -> usize {
        self.storage.capacity()
    }

    /// Get our length
    ///
    /// Where length is the number of elements that we are currently holding.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Get an element for reading
    ///
    /// Get the element at index. Returns `None` if the index is out of bounds.
    pub fn get(&self, index: usize) -> Option<&'a T> {
        if index >= self.len {
            None
        } else {
            // Safety: above we validate that index is within the bounds of our
            // allocated memory.
            let elt = unsafe { &*self.storage.as_ptr().offset(index as isize) };

            Some(elt)
        }
    }

    /// Get an element for writing
    ///
    /// Get the element at index. Returns `None` if the index is out of bounds.
    pub fn get_mut(&mut self, index: usize) -> Option<&'a mut T> {
        if index >= self.len {
            None
        } else {
            // Safety: above we validate that index is within the bounds of our
            // allocated memory.
            let elt = unsafe { &mut *self.storage.as_mut_ptr().offset(index as isize) };

            // Be sure to return an owned instance for mutation.
            elt.to_owned();

            Some(elt)
        }
    }

    /// Add an element to the collection
    ///
    /// Unless we need to reallocate capacity, this is a fast operation. If we
    /// do need to reallocate, it's a bit slower because we have to copy our
    /// memory to a new location.
    pub fn add(&mut self, value: T) {
        if self.len == self.capacity() {
            self.add_capacity();
        }

        let t = XuderCow::new_owned(value);

        // Safety: the buffer has capacity and our length is within that
        // capacity. We update our length immediately after the unsafe block.
        unsafe {
            let ptr = self.storage.as_mut_ptr().offset(self.len as isize);
            core::ptr::write(ptr, t);
        }

        self.len += 1;
    }

    // TODO: Double check this lifetime on self. It works, but is it right?
    pub fn iter(&'a self) -> Iter<'a, T> {
        Iter {
            index: 0,
            storage: &self,
        }
    }

    pub fn iter_mut(&'a mut self) -> IterMut<'a, T> {
        IterMut {
            index: 0,
            storage: self,
        }
    }

    pub fn is_owned(&self, index: usize) -> Option<bool> {
        if let Some(cow) = self.get_cow(index) {
            Some(cow.is_owned())
        } else {
            None
        }
    }
}

// Private methods
impl<'a, T> CowVec<'a, T>
where
    T: Clone,
{
    fn get_cow(&self, index: usize) -> Option<&XuderCow<'a, T>> {
        if index >= self.len {
            None
        } else {
            // Safety: We've checked the bounds above.
            Some(unsafe { &*self.storage.as_ptr().offset(index as isize) })
        }
    }

    fn add_capacity(&mut self) {
        let mut new_capacity = self.capacity() * 2;
        if new_capacity == 0 {
            // We'll pick a rando capacity if the storage is empty
            new_capacity = 10;
        }

        // We don't know where the extra capacity goes, so we start over with
        // a new `Vec`.
        let mut new = Vec::with_capacity(new_capacity);
        // Safety: we are copying self.len items to a fresh budder. self.len
        // is properly maintained.
        unsafe {
            core::ptr::copy_nonoverlapping(self.storage.as_ptr(), new.as_mut_ptr(), self.len);
        }
        self.storage = new;
    }
}

impl<'a, T> FromIterator<&'a T> for CowVec<'a, T>
where
    T: Clone,
{
    fn from_iter<I: IntoIterator<Item = &'a T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (low, high) = iter.size_hint();
        let size = if let Some(high) = high { high } else { low };
        let mut storage: Vec<XuderCow<T>> = Vec::with_capacity(size);

        let mut len = 0;
        for i in iter {
            // Safety: `len` is a valid index into self.storage
            unsafe {
                let ptr = storage.as_mut_ptr().offset(len as isize);
                core::ptr::write(ptr, XuderCow::new_ref(i));
            }
            len += 1;
        }

        CowVec { storage, len }
    }
}

impl<'a, T> IntoIterator for &'a CowVec<'a, T>
where
    T: Clone,
{
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub struct Iter<'a, T>
where
    T: Clone,
{
    index: usize,
    storage: &'a CowVec<'a, T>,
}

impl<'a, T> Iterator for Iter<'a, T>
where
    T: Clone,
{
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.storage.len() {
            None
        } else {
            self.index += 1;
            self.storage.get(self.index - 1)
        }
    }
}

pub struct IterMut<'a, T>
where
    T: Clone,
{
    index: usize,
    storage: &'a mut CowVec<'a, T>,
}

impl<'a, T> Iterator for IterMut<'a, T>
where
    T: Clone,
{
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.storage.len() {
            None
        } else {
            self.index += 1;
            self.storage.get_mut(self.index - 1)
        }
    }
}

impl<'a, T> core::ops::Index<usize> for CowVec<'a, T>
where
    T: Clone,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.len {
            panic!("Index out of bounds: {}. Length is {}.", index, self.len);
        }
        // Safety: We panic rather than read outside of the bounds.
        unsafe { &*self.storage.as_ptr().offset(index as isize) }
    }
}

impl<'a, T> core::ops::IndexMut<usize> for CowVec<'a, T>
where
    T: Clone,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index >= self.len {
            panic!("Index out of bounds: {}. Length is {}.", index, self.len);
        }
        // Safety: We panic rather than read outside of the bounds.
        unsafe { &mut *self.storage.as_mut_ptr().offset(index as isize) }
    }
}

/// I could make this one byte if I went with unsafe code and hijacked the LSB
/// of the pointer/ref. But ~~one vs two bytes~~ eight vs sixteen for what I'm
/// using it for seems unnecessary. That said, there's the part of me that
/// thinks about it in terms of ~~1 vs 2~~ 4 vs 8 is 100% bigger (50% savings).
/// So I may hack on this.
#[derive(Clone, Debug, PartialEq)]
enum XuderCow<'a, T>
where
    T: 'a + Clone,
{
    Owned(Box<T>),
    Borrowed(&'a T),
}

impl<'a, T> Deref for XuderCow<'a, T>
where
    T: Clone,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(t) => &t,
            Self::Borrowed(t) => *t,
        }
    }
}

impl<'a, T> DerefMut for XuderCow<'a, T>
where
    T: Clone,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Owned(t) => t,
            Self::Borrowed(t) => {
                // I just love this trick.
                let baz = &mut Self::Owned(Box::new((*t).clone()));
                core::mem::swap(self, baz);
                self
            }
        }
    }
}

impl<'a, T> XuderCow<'a, T>
where
    T: Clone,
{
    pub fn new_ref(value: &'a T) -> Self {
        XuderCow::Borrowed(value)
    }

    pub fn new_owned(value: T) -> Self {
        XuderCow::Owned(Box::new(value))
    }

    pub fn is_owned(&self) -> bool {
        match self {
            Self::Owned(_) => true,
            Self::Borrowed(_) => false,
        }
    }

    /// Make the referent owned
    ///
    /// If you want to explicitly make the referent a `Box<T>`, rather than a `&T`,
    /// then this is the function you'd call.
    pub fn to_owned(&mut self) -> &mut T {
        match self {
            Self::Owned(t) => t,
            Self::Borrowed(t) => {
                let baz = &mut Self::Owned(Box::new((*t).clone()));
                core::mem::swap(self, baz);
                self
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct Foo(usize);

    #[test]
    fn xuder_cow_basics() {
        // Create a reference
        let answer = Foo(42);
        // Now put it in a cow.
        let foo = XuderCow::new_ref(&answer);

        // They are the same.
        assert_eq!(answer, *foo);
        assert_eq!(*foo, Foo(42));
        assert!(!foo.is_owned());

        // foo should be immutable
        // foo.0 = 0;  <-- Does not compile.

        // Now create a mutable reference to foo, and make foo owned.
        // The compiler optimizes away the call to to_owned because the next
        // line calls deref_mut, which creates an owned copy as well.
        let mut bar = foo.to_owned();
        bar.0 = 27;

        // the original reference is intact, and our cow has a new value
        // note especially that `foo` is still a reference to the original
        // value
        assert_eq!(*foo, Foo(42)); // <--- original reference
        assert_eq!(*bar, Foo(27)); // <--- mutable reference to above
        assert_eq!(answer, Foo(42)); // <--- original owned Foo
        assert!(bar.is_owned()); // <--- cloned Fooo
    }

    #[test]
    fn from_iter() {
        let v = vec![Foo(42), Foo(27)];
        let cow: CowVec<Foo> = v.iter().map(|a| a).collect();

        // check capacity and length. Ensure underlying storage is empty,
        assert_eq!(cow.capacity(), 2);
        assert_eq!(cow.len(), 2);
        assert_eq!(cow.storage.len(), 0);

        // Check each element in the list for equality, and that it's a reference.
        cow.storage.iter().enumerate().for_each(|(i, e)| {
            assert!(!e.is_owned());
            assert_eq!(**e, v[i as usize]);
        });
    }

    #[test]
    fn new_from_vec_ref() {
        let v = vec![Foo(42), Foo(27)];
        let cow = CowVec::new(&v);
        assert_eq!(cow.capacity(), 2);
        assert_eq!(cow.len(), 2);
        assert_eq!(cow.storage.len(), 0);

        cow.storage.iter().enumerate().for_each(|(i, e)| {
            assert!(!e.is_owned());
            assert_eq!(**e, v[i as usize]);
        });
    }

    #[test]
    fn get_element() {
        let v = vec![Foo(42), Foo(27)];
        let cow = CowVec::new(&v);

        let first = cow.get(0);
        assert!(first.is_some());
        assert_eq!(first, v.get(0));

        // Try to go beyond the end of the collection
        let third = cow.get(3);
        assert!(third.is_none());
    }

    #[test]
    fn get_element_via_index() {
        let v = vec![Foo(42), Foo(27)];
        let cow = CowVec::new(&v);

        assert_eq!(&cow[0], &v[0]);
        assert_eq!(&cow[1], &v[1]);
    }

    #[test]
    #[should_panic]
    fn index_out_of_bounds() {
        let v = vec![Foo(42), Foo(27)];
        let cow = CowVec::new(&v);

        // Try to go beyond the end of the collection
        let _ = &cow[2];
    }

    #[test]
    fn get_mut_element_via_index() {
        let v = vec![Foo(42), Foo(27)];
        let mut cow = CowVec::new(&v);

        {
            let mut a = &mut cow[0];
            a.0 += 27;
        }
        {
            let mut b = &mut cow[1];
            b.0 += 42;
        }

        assert_eq!(&cow[0], &cow[1]);
    }

    #[test]
    #[should_panic]
    fn mut_index_out_of_bounds() {
        let v = vec![Foo(42), Foo(27)];
        let mut cow = CowVec::new(&v);

        // Try to go beyond the end of the collection
        let _ = &mut cow[2];
    }

    #[test]
    fn get_mut_element() {
        let v = vec![Foo(42), Foo(27)];
        let mut cow = CowVec::new(&v);

        {
            let first = cow.get_mut(0).unwrap();
            first.0 += 27;
        }

        assert_eq!(*cow.get(0).unwrap(), Foo(42 + 27));
    }

    #[test]
    fn check_bounds() {
        let v = vec![Foo(42), Foo(27)];
        let mut cow = CowVec::new(&v);

        assert!(cow.get(2).is_none());
        assert!(cow.get_mut(2).is_none());
        assert!(cow.is_owned(2).is_none());
    }

    #[test]
    fn add_element() {
        let v = vec![Foo(42), Foo(27)];
        let mut cow = CowVec::new(&v);

        cow.add(Foo(42 + 27));
        assert_eq!(cow.len(), 3);

        let elt = cow.get(2);
        assert!(elt.is_some());
        assert_eq!(elt.unwrap(), &Foo(42 + 27));

        assert!(!cow.is_owned(0).unwrap());
        assert!(!cow.is_owned(1).unwrap());
        assert!(cow.is_owned(2).unwrap());
    }

    #[test]
    fn test_add_capacity() {
        // This is deviant, and is not recommended.
        let v = vec![];
        let mut cow = CowVec::new(&v);
        assert_eq!(cow.capacity(), 0);

        cow.add(Foo(42 + 27));
        assert_eq!(cow.capacity(), 10);
    }

    #[test]
    fn mut_iteration() {
        let v = vec![Foo(42), Foo(27)];
        let mut cow = CowVec::new(&v);

        let ci: Vec<&mut Foo> = cow
            .iter_mut()
            .map(|e| {
                e.0 += 10;
                e
            })
            .collect();

        let mut v1 = v.clone();
        let vi: Vec<&mut Foo> = v1
            .iter_mut()
            .map(|e| {
                e.0 += 10;
                e
            })
            .collect();
        assert_eq!(ci, vi);
    }

    #[test]
    fn test_into_iter() {
        let v = vec![Foo(42), Foo(27)];
        let cow = CowVec::new(&v);

        for (i, e) in cow.into_iter().enumerate() {
            assert_eq!(*e, v[i]);
        }
    }

    #[test]
    fn iteration() {
        let v = vec![Foo(42), Foo(27)];
        let cow = CowVec::new(&v);

        let ci = cow.iter().fold(0, |acc, e| acc + e.0);
        let vi = v.iter().fold(0, |acc, e| acc + e.0);
        assert_eq!(ci, vi);

        println!(
            "size of Foo {}, size of XuderCow<Foo> {}",
            core::mem::size_of::<Foo>(),
            core::mem::size_of::<XuderCow<Foo>>()
        );
    }
}
