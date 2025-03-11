instance : Subsingleton (Path PUnit.unit PUnit.unit) :=
                 /-
                   x y : Path PUnit.unit PUnit.unit
                   ⊢ Eq x y
                 -/
  ⟨fun x y => by ext⟩
                 /-
                   🎉 no goals
                 -/


instance {x y : FundamentalGroupoid PUnit} : Subsingleton (x ⟶ y) := by
  /-
    x y : FundamentalGroupoid PUnit.{u_1 + 1}
    ⊢ Subsingleton (Quiver.Hom x y)
  -/
  convert_to Subsingleton (Path.Homotopic.Quotient PUnit.unit PUnit.unit)
  /-
    x y : FundamentalGroupoid PUnit.{u_1 + 1}
    ⊢ Subsingleton (Path.Homotopic.Quotient PUnit.unit PUnit.unit)
  -/
  apply Quotient.instSubsingletonQuotient
  /-
    🎉 no goals
  -/


/-- Equivalence of groupoids between fundamental groupoid of punit and punit -/
@[simps]
def punitEquivDiscretePUnit : FundamentalGroupoid PUnit.{u + 1} ≌ Discrete PUnit.{v + 1} where
  functor := Functor.star _
  inverse := (CategoryTheory.Functor.const _).obj ⟨PUnit.unit⟩
             /-
               ⊢ ∀ {X Y : FundamentalGroupoid PUnit.{u + 1}} (f : Quiver.Hom X Y), Eq (Catego …
             -/
  unitIso := NatIso.ofComponents (fun _ => Iso.refl _)
             /-
               🎉 no goals
             -/
  counitIso := Iso.refl _


