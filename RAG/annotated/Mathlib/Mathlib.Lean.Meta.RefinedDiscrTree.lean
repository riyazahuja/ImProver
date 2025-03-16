/-- If `vs` contains an element `v'` such that `v == v'`, then replace `v'` with `v`.
Otherwise, push `v`.
See issue https://github.com/leanprover-community/mathlib4/pull/2155
Recall that `BEq α` may not be Lawful.
-/
private def insertInArray [BEq α] (vs : Array α) (v : α) : Array α :=
  loop 0
where
  loop (i : Nat) : Array α :=
    if h : i < vs.size then
      if v == vs[i] then
        /-
          α : Type ?u.8
          inst✝ : BEq α
          vs : Array α
          v : α
          i : Nat
          h : LT.lt i vs.size
          ⊢ LT.lt i vs.size
        -/
        vs.set i v
        /-
          🎉 no goals
        -/
      else
        loop (i+1)
    else
      vs.push v


/-- Insert the value `v` at index `keys : Array Key` in a `Trie`. -/
partial def insertInTrie [BEq α] (keys : Array Key) (v : α) (i : Nat) : Trie α → Trie α
  | .node cs =>
      let k := keys[i]!
      let c := Id.run <| cs.binInsertM
        (fun a b => a.1 < b.1)
        (fun (k', s) => (k', insertInTrie keys v (i+1) s))
        (fun _ => (k, Trie.singleton keys v (i+1)))
        (k, default)
      .node c
  | .values vs =>
      .values (insertInArray vs v)
  | .path ks c => Id.run do
    for n in [:ks.size] do
      let k1 := keys[i+n]!
      let k2 := ks[n]!
      if k1 != k2 then
        let shared := ks[:n]
        let rest := ks[n+1:]
        return .mkPath shared (.mkNode2 k1 (.singleton keys v (i+n+1)) k2 (.mkPath rest c))
    return .path ks (insertInTrie keys v (i + ks.size) c)


/-- Insert the value `v` at index `keys : Array Key` in a `RefinedDiscrTree`.

Warning: to account for η-reduction, an entry may need to be added at multiple indexes,
so it is recommended to use `RefinedDiscrTree.insert` for insertion. -/
def insertInRefinedDiscrTree [BEq α] (d : RefinedDiscrTree α) (keys : Array Key) (v : α) :
    RefinedDiscrTree α :=
  let k := keys[0]!
  match d.root.find? k with
  | none =>
    let c := .singleton keys v 1
    { root := d.root.insert k c }
  | some c =>
    let c := insertInTrie keys v 1 c
    { root := d.root.insert k c }


/-- Insert the value `v` at index `e : DTExpr` in a `RefinedDiscrTree`.

Warning: to account for η-reduction, an entry may need to be added at multiple indexes,
so it is recommended to use `RefinedDiscrTree.insert` for insertion. -/
def insertDTExpr [BEq α] (d : RefinedDiscrTree α) (e : DTExpr) (v : α) : RefinedDiscrTree α :=
  insertInRefinedDiscrTree d e.flatten v


/-- Insert the value `v` at index `e : Expr` in a `RefinedDiscrTree`.
The argument `fvarInContext` allows you to specify which free variables in `e` will still be
in the context when the `RefinedDiscrTree` is being used for lookup.
It should return true only if the `RefinedDiscrTree` is built and used locally.

if `onlySpecific := true`, then we filter out the patterns `*` and `Eq * * *`. -/
def insert [BEq α] (d : RefinedDiscrTree α) (e : Expr) (v : α)
    (onlySpecific : Bool := true) (fvarInContext : FVarId → Bool := fun _ => false) :
    MetaM (RefinedDiscrTree α) := do
  let keys ← mkDTExprs e onlySpecific fvarInContext
  return keys.foldl (insertDTExpr · · v) d


/-- Insert the value `vLhs` at index `lhs`, and if `rhs` is indexed differently, then also
insert the value `vRhs` at index `rhs`. -/
def insertEqn [BEq α] (d : RefinedDiscrTree α) (lhs rhs : Expr) (vLhs vRhs : α)
    (onlySpecific : Bool := true) (fvarInContext : FVarId → Bool := fun _ => false) :
    MetaM (RefinedDiscrTree α) := do
  let keysLhs ← mkDTExprs lhs onlySpecific fvarInContext
  let keysRhs ← mkDTExprs rhs onlySpecific fvarInContext
  let d := keysLhs.foldl (insertDTExpr · · vLhs) d
  if @List.beq _ ⟨DTExpr.eqv⟩ keysLhs keysRhs then
    return d
  else
    return keysRhs.foldl (insertDTExpr · · vRhs) d




/-- Apply a monadic function to the array of values at each node in a `RefinedDiscrTree`. -/
partial def Trie.mapArraysM (t : RefinedDiscrTree.Trie α) (f : Array α → m (Array β)) :
    m (Trie β) := do
  match t with
  | .node children =>
    return .node (← children.mapM fun (k, t') => do pure (k, ← t'.mapArraysM f))
  | .values vs =>
    return .values (← f vs)
  | .path ks c =>
    return .path ks (← c.mapArraysM f)


/-- Apply a monadic function to the array of values at each node in a `RefinedDiscrTree`. -/
def mapArraysM (d : RefinedDiscrTree α) (f : Array α → m (Array β)) : m (RefinedDiscrTree β) :=
  return { root := ← d.root.mapM (·.mapArraysM f) }


/-- Apply a function to the array of values at each node in a `RefinedDiscrTree`. -/
def mapArrays (d : RefinedDiscrTree α) (f : Array α → Array β) : RefinedDiscrTree β :=
  d.mapArraysM (m := Id) f


