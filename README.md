# cow-vec

CowVec: A vector of CoW elements
The intended purpose of this implementation is to provide a means to create
cheap references to the elements in another `Vec`. As long as there is no
mutation, access is fast, and the memory cost is low.

If you want to mutate an element, we utilize XuderCow to create a new Boxed
element which is mutable, and does not effect the other `Vec`.

This implementation uses unsafe code to store the elements. We allocate
memory by means of a `Vec`, who's len is always zero.

I wrote this partly for fun, but mostly because I needed a collection that
I could implement mutable iterators on. I was also hoping that I could get
mutable and immutable references to elements at the same time. Only because
the [XuderCow] can change its referent to a `Box<T>` without invalidating the
references that were pointing at the `&T`. Maybe I can do something tricky
with lifetimes.
