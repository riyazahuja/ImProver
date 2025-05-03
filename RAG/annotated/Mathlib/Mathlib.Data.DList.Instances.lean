/-- The natural equivalence between lists and difference lists, using
`DList.ofList` and `DList.toList`. -/
def DList.listEquivDList : List α ≃ DList α where
  toFun := DList.ofList
  invFun := DList.toList
  left_inv _ := DList.toList_ofList _
  right_inv _ := DList.ofList_toList _


instance : Traversable DList :=
  Equiv.traversable DList.listEquivDList


instance : LawfulTraversable DList :=
  Equiv.isLawfulTraversable DList.listEquivDList


instance {α} : Inhabited (DList α) :=
  ⟨DList.empty⟩


