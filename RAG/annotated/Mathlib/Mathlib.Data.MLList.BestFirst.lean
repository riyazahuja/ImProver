/-- A node in a `BestFirstQueue`. -/
structure BestFirstNode {α : Sort*} {ω : Type*} (prio : α → Thunk ω) (ε : α → Type) where
  /-- The data to store at a node, from which we can calculate a priority using `prio`. -/
  key : α
  /-- An estimator for the priority of the key.
  (We will assume we have `[∀ a : α, Estimator (prio a) (ε a)]`.) -/
  estimator : ε key


/-- Calculate the current best lower bound for the priority of a node. -/
def BestFirstNode.estimate (n : BestFirstNode prio ε) : ω := bound (prio n.key) n.estimator


instance [Ord ω] [Ord α] : Ord (BestFirstNode prio ε) where
  compare :=
    compareLex
      (compareOn BestFirstNode.estimate)
      (compareOn BestFirstNode.key)


set_option linter.unusedVariables false in
variable (prio ε m β) [Ord ω] [Ord α] in
/-- A queue of `MLList m β`s, lazily prioritized by lower bounds. -/
@[nolint unusedArguments]
def BestFirstQueue (maxSize : Option Nat) := RBMap (BestFirstNode prio ε) (MLList m β) compare


/--
Add a new `MLList m β` to the `BestFirstQueue`, and if this takes the size above `maxSize`,
eject a `MLList` from the tail of the queue.
-/
-- Note this ejects the element with the greatest estimated priority,
-- not necessarily the greatest priority!
def insertAndEject
    (q : BestFirstQueue prio ε m β maxSize) (n : BestFirstNode prio ε) (l : MLList m β) :
    BestFirstQueue prio ε m β maxSize :=
  match maxSize with
  | none => q.insert n l
  | some max =>
    if q.size < max then
      q.insert n l
    else
      match q.max? with
      | none => RBMap.empty
      | some m => q.insert n l |>.erase m.1


/--
By improving priority estimates as needed, and permuting elements,
ensure that the first element of the queue has the greatest priority.
-/
partial def ensureFirstIsBest (q : BestFirstQueue prio ε m β maxSize) :
    m (BestFirstQueue prio ε m β maxSize) := do
  let s := @toStream (RBMap _ _ _) _ _ q
  match s.next? with
  | none =>
    -- The queue is empty, nothing to do.
    return q
  | some ((n, l), s') => match s'.next? with
    | none => do
      -- There's only one element in the queue, no reordering necessary.
      return q
    | some ((m, _), _) =>
      -- `n` is the first index, `m` is the second index.
      -- We need to improve our estimate of the priority for `n` to make sure
      -- it really should come before `m`.
      match improveUntil (prio n.key) (m.estimate < ·) n.estimator with
      | .error none =>
        -- If we couldn't improve the estimate at all, it is exact, and hence the best element.
        return q
      | .error (some e') =>
        -- If we improve the estimate, but it is still at most the estimate for `m`,
        -- this is the best element, so all we need to do is store the updated estimate.
        return q.erase n |>.insert ⟨n.key, e'⟩ l
      | .ok e' =>
        -- If we improved the estimate and it becomes greater than the estimate for `m`,
        -- we re-insert `n` with its new estimate, and then try again.
        ensureFirstIsBest (q.erase n |>.insert ⟨n.key, e'⟩ l)


/--
Pop a `β` off the `MLList m β` with lowest priority,
also returning the index in `α` and the current best lower bound for its priority.
This may require improving estimates of priorities and shuffling the queue.
-/
partial def popWithBound (q : BestFirstQueue prio ε m β maxSize) :
    m (Option (((a : α) × (ε a) × β) × BestFirstQueue prio ε m β maxSize)) := do
  let q' ← ensureFirstIsBest q
  match q'.min? with
  | none =>
    -- The queue is empty, nothing to return.
    return none
  | some (n, l) =>
    match ← l.uncons with
    | none =>
      -- The `MLList` associated to `n` was actually empty, so we remove `n` and try again.
      popWithBound (q'.erase n)
    | some (b, l') =>
      -- Return the initial element `b` along with the current estimator,
      -- and replace the `MLList` associated with `n` with its tail.
      return some (⟨n.key, n.estimator, b⟩, q'.modify n fun _ => l')


/--
Pop a `β` off the `MLList m β` with lowest priority,
also returning the index in `α` and the value of the current best lower bound for its priority.
-/
def popWithPriority (q : BestFirstQueue prio ε m β maxSize) :
    m (Option (((α × ω) × β) × BestFirstQueue prio ε m β maxSize)) := do
  match ← q.popWithBound with
  | none => pure none
  | some (⟨a, e, b⟩, q') => pure (some (((a, bound (prio a) e), b), q'))


/--
Pop a `β` off the `MLList m β` with lowest priority.
-/
def pop (q : BestFirstQueue prio ε m β maxSize) :
    m (Option ((α × β) × BestFirstQueue prio ε m β maxSize)) := do
  match ← q.popWithBound with
  | none => pure none
  | some (⟨a, _, b⟩, q') => pure (some ((a, b), q'))


/--
Convert a `BestFirstQueue` to a `MLList ((α × ω) × β)`, by popping off all elements,
recording also the values in `ω` of the best current lower bounds.
-/
-- This is not used in the algorithms below, but can be useful for debugging.
partial def toMLListWithPriority (q : BestFirstQueue prio ε m β maxSize) : MLList m ((α × ω) × β) :=
  .squash fun _ => do
    match ← q.popWithPriority with
    | none => pure .nil
    | some (p, q') => pure <| MLList.cons p q'.toMLListWithPriority


/--
Convert a `BestFirstQueue` to a `MLList (α × β)`, by popping off all elements.
-/
def toMLList (q : BestFirstQueue prio ε m β maxSize) : MLList m (α × β) :=
  q.toMLListWithPriority.map fun t => (t.1.1, t.2)


/--
Core implementation of `bestFirstSearch`, that works by iteratively updating an internal state,
consisting of a priority queue of `MLList m α`.

At each step we pop an element off the queue,
compute its children (lazily) and put these back on the queue.
-/
def impl (maxSize : Option Nat) (f : α → MLList m α) (a : α) : MLList m α :=
  let init : BestFirstQueue prio ε m α maxSize := RBMap.single ⟨a, ⊥⟩ (f a)
  cons a (iterate go |>.runState' init)
where
  /-- A single step of the best first search.
  Pop an element, and insert its children back into the queue,
  with a trivial estimator for their priority. -/
  go : StateT (BestFirstQueue prio ε m α maxSize) m α := fun s => do
  match ← s.pop with
    | none => failure
    | some ((_, b), s') => pure (b, s'.insertAndEject ⟨b, ⊥⟩ (f b))


/--
Wrapper for `impl` implementing the `maxDepth` option.
-/
def implMaxDepth (maxSize : Option Nat) (maxDepth : Option Nat) (f : α → MLList m α) (a : α) :
    MLList m α :=
  match maxDepth with
  | none => impl prio ε maxSize f a
  | some max =>
    let f' : α ×ₗ Nat → MLList m (α × Nat) := fun ⟨a, n⟩ =>
      if max < n then
        nil
      else
        (f a).map fun a' => (a', n + 1)
    impl (fun p : α ×ₗ Nat => prio p.1) (fun p : α ×ₗ Nat => ε p.1) maxSize f' (a, 0) |>.map (·.1)


/--
A lazy list recording the best first search of a graph generated by a function
`f : α → MLList m α`.

We maintain a priority queue of visited-but-not-exhausted nodes,
and at each step take the next child of the highest priority node in the queue.

The option `maxDepth` limits the search depth.

The option `maxQueued` bounds the size of the priority queue,
discarding the lowest priority nodes as needed.
This implements a "beam" search, which may be incomplete but uses bounded memory.

The option `removeDuplicates` keeps an `RBSet` of previously visited nodes.
Otherwise, if the graph is not a tree then nodes will be visited multiple times.

This version allows specifying a custom priority function `prio : α → Thunk ω`
along with estimators `ε : α → Type` equipped with `[∀ a, Estimator (prio a) (ε a)]`
that control the behaviour of the priority queue.
This function returns values `a : α` that have
the lowest possible `prio a` amongst unvisited neighbours of visited nodes,
but lazily estimates these priorities to avoid unnecessary computations.
-/
def bestFirstSearchCore (f : α → MLList m α) (a : α)
    (β : Type _) [Ord β] (removeDuplicatesBy? : Option (α → β) := none)
    (maxQueued : Option Nat := none) (maxDepth : Option Nat := none)  :
    MLList m α :=
  match removeDuplicatesBy? with
  | some g =>
    let f' : α → MLList (StateT (RBSet β compare) m) α := fun a =>
      (f a).liftM >>= fun a' => do
        let b := g a'
        guard !(← get).contains b
        modify fun s => s.insert b
        pure a'
    implMaxDepth prio ε maxQueued maxDepth f' a |>.runState' (RBSet.empty.insert (g a))
  | none =>
    implMaxDepth prio ε maxQueued maxDepth f a


/-- A local instance that enables using "the actual value" as a priority estimator,
for simple use cases. -/
local instance instOrderBotEq {a : α} : OrderBot { x : α // x = a } where
  bot := ⟨a, rfl⟩
               /-
                 m : Type → Type
                 α : Type
                 inst✝² : Monad m
                 inst✝¹ : Alternative m
                 inst✝ : LinearOrder α
                 a : α
                 ⊢ ∀ (a_1 : Subtype fun x => Eq x a), LE.le Bot.bot a_1
               -/
  bot_le := by aesop
               /-
                 🎉 no goals
               -/


/--
A lazy list recording the best first search of a graph generated by a function
`f : α → MLList m α`.

We maintain a priority queue of visited-but-not-exhausted nodes,
and at each step take the next child of the highest priority node in the queue.

The option `maxDepth` limits the search depth.

The option `maxQueued` bounds the size of the priority queue,
discarding the lowest priority nodes as needed.
This implements a "beam" search, which may be incomplete but uses bounded memory.

The option `removeDuplicates` keeps an `RBSet` of previously visited nodes.
Otherwise, if the graph is not a tree then nodes will be visited multiple times.

This function returns values `a : α` that are least in the `[LinearOrder α]`
amongst unvisited neighbours of visited nodes.
-/
-- Although the core implementation lazily computes estimates of priorities,
-- this version does not take advantage of those features.
def bestFirstSearch (f : α → MLList m α) (a : α)
    (maxQueued : Option Nat := none) (maxDepth : Option Nat := none) (removeDuplicates := true) :
    MLList m α :=
  bestFirstSearchCore Thunk.pure (fun a : α => { x // x = a }) f a
    (β := α) (removeDuplicatesBy? := if removeDuplicates then some id else none)
    maxQueued maxDepth

