lemma eq_liftCycles_homologyπ_up_to_refinements {A : C} {i : ι} (γ : A ⟶ K.homology i)
    (j : ι) (hj : c.next i = j) :
    ∃ (A' : C) (π : A' ⟶ A) (_ : Epi π) (z : A' ⟶ K.X i) (hz : z ≫ K.d i j = 0),
      π ≫ γ = K.liftCycles z j hj hz ≫ K.homologyπ i := by
  /-
    C : Type u_1
    ι : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Abelian C
    c : ComplexShape ι
    K : HomologicalComplex C c
    A : C
    i : ι
    γ : Quiver.Hom A (K.homology i)
    j : ι
    hj : Eq (c.next i) j
    ⊢ Exists fun A' => Exists fun π => Exists fun x => Exists fun z => Exists fun  …
  -/
  subst hj
  /-
    C : Type u_1
    ι : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Abelian C
    c : ComplexShape ι
    K : HomologicalComplex C c
    A : C
    i : ι
    γ : Quiver.Hom A (K.homology i)
    ⊢ Exists fun A' => Exists fun π => Exists fun x => Exists fun z => Exists fun  …
  -/
  exact (K.sc i).eq_liftCycles_homologyπ_up_to_refinements γ
  /-
    🎉 no goals
  -/


