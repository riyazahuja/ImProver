/-- The list of keys in a `HashMap`. -/
@[deprecated "This declaration is unused in Mathlib: if you need it, \
  please file an issue in the Batteries repository." (since := "2024-06-12")]
def keys (m : HashMap α β) : List α :=
  m.fold (fun ks k _ => k :: ks) []


/-- The list of values in a `HashMap`. -/
@[deprecated "This declaration is unused in Mathlib: if you need it, \
  please file an issue in the Batteries repository." (since := "2024-06-12")]
def values (m : HashMap α β) : List β :=
  m.fold (fun vs _ v => v :: vs) []


/-- Add a value to a `HashMap α (List β)` viewed as a multimap. -/
@[deprecated "This declaration is unused in Mathlib: if you need it, \
  please file an issue in the Batteries repository." (since := "2024-06-12")]
def consVal (self : HashMap α (List β)) (a : α) (b : β) : HashMap α (List β) :=
  match self.find? a with
  | none => self.insert a [b]
  | some L => self.insert a (b::L)


/-- Insert all elements of a list into an `RBSet`. -/
@[deprecated "This declaration is unused in Mathlib: if you need it, \
  please file an issue in the Batteries repository." (since := "2024-06-12")]
def insertList {cmp} (m : RBSet α cmp) (L : List α) : RBSet α cmp :=
  L.foldl (fun m a => m.insert a) m


