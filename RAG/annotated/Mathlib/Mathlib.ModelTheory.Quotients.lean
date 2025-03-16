/-- A prestructure is a first-order structure with a `Setoid` equivalence relation on it,
  such that quotienting by that equivalence relation is still a structure. -/
class Prestructure (s : Setoid M) where
  toStructure : L.Structure M
  fun_equiv : ∀ {n} {f : L.Functions n} (x y : Fin n → M), x ≈ y → funMap f x ≈ funMap f y
  rel_equiv : ∀ {n} {r : L.Relations n} (x y : Fin n → M) (_ : x ≈ y), RelMap r x = RelMap r y


instance quotientStructure : L.Structure (Quotient s) where
  funMap {n} f x :=
    Quotient.map (@funMap L M ps.toStructure n f) Prestructure.fun_equiv (Quotient.finChoice x)
  RelMap {n} r x :=
    Quotient.lift (@RelMap L M ps.toStructure n r) Prestructure.rel_equiv (Quotient.finChoice x)


theorem funMap_quotient_mk' {n : ℕ} (f : L.Functions n) (x : Fin n → M) :
    (funMap f fun i => (⟦x i⟧ : Quotient s)) = ⟦@funMap _ _ ps.toStructure _ f x⟧ := by
  change
    Quotient.map (@funMap L M ps.toStructure n f) Prestructure.fun_equiv (Quotient.finChoice _) =
      _
  /-
    L : FirstOrder.Language
    M : Type u_1
    s : Setoid M
    ps : L.Prestructure s
    n : Nat
    f : L.Functions n
    x : Fin n → M
    ⊢ Eq (Quotient.map (FirstOrder.Language.Structure.funMap f) ⋯ (Quotient.finCho …
  -/
  rw [Quotient.finChoice_eq, Quotient.map_mk]
  /-
    🎉 no goals
  -/


theorem relMap_quotient_mk' {n : ℕ} (r : L.Relations n) (x : Fin n → M) :
    (RelMap r fun i => (⟦x i⟧ : Quotient s)) ↔ @RelMap _ _ ps.toStructure _ r x := by
  change
    Quotient.lift (@RelMap L M ps.toStructure n r) Prestructure.rel_equiv (Quotient.finChoice _) ↔
      _
  /-
    L : FirstOrder.Language
    M : Type u_1
    s : Setoid M
    ps : L.Prestructure s
    n : Nat
    r : L.Relations n
    x : Fin n → M
    ⊢ Iff (Quotient.lift (FirstOrder.Language.Structure.RelMap r) ⋯ (Quotient.finC …
  -/
  rw [Quotient.finChoice_eq, Quotient.lift_mk]
  /-
    🎉 no goals
  -/


theorem Term.realize_quotient_mk' {β : Type*} (t : L.Term β) (x : β → M) :
    (t.realize fun i => (⟦x i⟧ : Quotient s)) = ⟦@Term.realize _ _ ps.toStructure _ x t⟧ := by
  induction t with
  | var => rfl
  | func _ _ ih => simp only [ih, funMap_quotient_mk', Term.realize]


