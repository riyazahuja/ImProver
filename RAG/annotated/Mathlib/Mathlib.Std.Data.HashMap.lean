/-- Apply a function to the values of a hash map. -/
def mapVal (f : α → β → γ) (m : HashMap α β) : HashMap α γ :=
  m.fold (fun acc k v => acc.insert k (f k v)) HashMap.empty


