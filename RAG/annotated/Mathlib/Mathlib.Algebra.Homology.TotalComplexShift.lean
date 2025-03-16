/-- The shift on bicomplexes obtained by shifting the first indices (and changing the
sign of differentials). -/
abbrev shiftFunctor₁ (x : ℤ) :
    HomologicalComplex₂ C (up ℤ) (up ℤ) ⥤ HomologicalComplex₂ C (up ℤ) (up ℤ) :=
  shiftFunctor _ x


/-- The shift on bicomplexes obtained by shifting the second indices (and changing the
sign of differentials). -/
abbrev shiftFunctor₂ (y : ℤ) :
    HomologicalComplex₂ C (up ℤ) (up ℤ) ⥤ HomologicalComplex₂ C (up ℤ) (up ℤ) :=
  (shiftFunctor _ y).mapHomologicalComplex _


/-- The isomorphism `(((shiftFunctor₁ C x).obj K).X a).X b ≅ (K.X a').X b` when `a' = a + x`. -/
def shiftFunctor₁XXIso (a x a' : ℤ) (h : a' = a + x) (b : ℤ) :
                                                                        /-
                                                                          C : Type u_1
                                                                          inst✝¹ : CategoryTheory.Category.{?u.2837, u_1} C
                                                                          inst✝ : CategoryTheory.Preadditive C
                                                                          K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
                                                                          f : Quiver.Hom K L
                                                                          a x a' : Int
                                                                          h : Eq a' (HAdd.hAdd a x)
                                                                          b : Int
                                                                          ⊢ Eq ((((HomologicalComplex₂.shiftFunctor₁ C x).obj K).X a).X b) ((K.X a').X b)
                                                                        -/
    (((shiftFunctor₁ C x).obj K).X a).X b ≅ (K.X a').X b := eqToIso (by subst h; rfl)
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


/-- The isomorphism `(((shiftFunctor₂ C y).obj K).X a).X b ≅ (K.X a).X b'` when `b' = b + y`. -/
def shiftFunctor₂XXIso (a b y b' : ℤ) (h : b' = b + y) :
                                                                        /-
                                                                          C : Type u_1
                                                                          inst✝¹ : CategoryTheory.Category.{?u.3771, u_1} C
                                                                          inst✝ : CategoryTheory.Preadditive C
                                                                          K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
                                                                          f : Quiver.Hom K L
                                                                          a b y b' : Int
                                                                          h : Eq b' (HAdd.hAdd b y)
                                                                          ⊢ Eq ((((HomologicalComplex₂.shiftFunctor₂ C y).obj K).X a).X b) ((K.X a).X b')
                                                                        -/
    (((shiftFunctor₂ C y).obj K).X a).X b ≅ (K.X a).X b' := eqToIso (by subst h; rfl)
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


@[simp]
lemma shiftFunctor₁XXIso_refl (a b x : ℤ) :
    K.shiftFunctor₁XXIso a x (a + x) rfl b = Iso.refl _ := rfl


@[simp]
lemma shiftFunctor₂XXIso_refl (a b y : ℤ) :
    K.shiftFunctor₂XXIso a b y (b + y) rfl = Iso.refl _ := rfl


instance : ((shiftFunctor₁ C x).obj K).HasTotal (up ℤ) := fun n =>
  hasCoproduct_of_equiv_of_iso (K.toGradedObject.mapObjFun (π (up ℤ) (up ℤ) (up ℤ)) (n + x)) _
    { toFun := fun ⟨⟨a, b⟩, h⟩ => ⟨⟨a + x, b⟩, by
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{u_2, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
          f : Quiver.Hom K L
          x y : Int
          inst✝ : K.HasTotal (ComplexShape.up Int)
          n : Int
          x✝ : ↑(Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexSha …
          a b : Int
          h : Membership.mem (Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int …
          ⊢ Membership.mem (Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int)  …
        -/
        simp only [Set.mem_preimage, instTotalComplexShape_π, Set.mem_singleton_iff] at h ⊢
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{u_2, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
          f : Quiver.Hom K L
          x y : Int
          inst✝ : K.HasTotal (ComplexShape.up Int)
          n : Int
          x✝ : ↑(Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexSha …
          a b : Int
          h : Eq (HAdd.hAdd a b) n
          ⊢ Eq (HAdd.hAdd (HAdd.hAdd a x) b) (HAdd.hAdd n x)
        -/
        omega⟩
        /-
          🎉 no goals
        -/
      invFun := fun ⟨⟨a, b⟩, h⟩ => ⟨(a - x, b), by
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{u_2, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
          f : Quiver.Hom K L
          x y : Int
          inst✝ : K.HasTotal (ComplexShape.up Int)
          n : Int
          x✝ : ↑(Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexSha …
          a b : Int
          h : Membership.mem (Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int …
          ⊢ Membership.mem (Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int)  …
        -/
        simp only [Set.mem_preimage, instTotalComplexShape_π, Set.mem_singleton_iff] at h ⊢
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{u_2, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
          f : Quiver.Hom K L
          x y : Int
          inst✝ : K.HasTotal (ComplexShape.up Int)
          n : Int
          x✝ : ↑(Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexSha …
          a b : Int
          h : Eq (HAdd.hAdd a b) (HAdd.hAdd n x)
          ⊢ Eq (HAdd.hAdd (HSub.hSub a x) b) n
        -/
        omega⟩
        /-
          🎉 no goals
        -/
      left_inv := by
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{u_2, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
          f : Quiver.Hom K L
          x y : Int
          inst✝ : K.HasTotal (ComplexShape.up Int)
          n : Int
          ⊢ Function.LeftInverse (fun x_1 => HomologicalComplex₂.instHasTotalIntObjUpShi …
        -/
        rintro ⟨⟨a, b⟩, h⟩
        /-
          case mk.mk
          C : Type u_1
          inst✝² : CategoryTheory.Category.{u_2, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
          f : Quiver.Hom K L
          x y : Int
          inst✝ : K.HasTotal (ComplexShape.up Int)
          n a b : Int
          h : Membership.mem (Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int …
          ⊢ Eq ((fun x_1 => HomologicalComplex₂.instHasTotalIntObjUpShiftFunctor₁.match_ …
        -/
        ext
          /-
            case mk.mk.a.fst
            C : Type u_1
            inst✝² : CategoryTheory.Category.{u_2, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
            f : Quiver.Hom K L
            x y : Int
            inst✝ : K.HasTotal (ComplexShape.up Int)
            n a b : Int
            h : Membership.mem (Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int …
            ⊢ Eq (↑((fun x_1 => HomologicalComplex₂.instHasTotalIntObjUpShiftFunctor₁.matc …
          -/
        · dsimp
          /-
            case mk.mk.a.fst
            C : Type u_1
            inst✝² : CategoryTheory.Category.{u_2, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
            f : Quiver.Hom K L
            x y : Int
            inst✝ : K.HasTotal (ComplexShape.up Int)
            n a b : Int
            h : Membership.mem (Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int …
            ⊢ Eq (HSub.hSub (HAdd.hAdd a x) x) a
          -/
          omega
          /-
            🎉 no goals
          -/
          /-
            case mk.mk.a.snd
            C : Type u_1
            inst✝² : CategoryTheory.Category.{u_2, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
            f : Quiver.Hom K L
            x y : Int
            inst✝ : K.HasTotal (ComplexShape.up Int)
            n a b : Int
            h : Membership.mem (Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int …
            ⊢ Eq (↑((fun x_1 => HomologicalComplex₂.instHasTotalIntObjUpShiftFunctor₁.matc …
          -/
        · rfl
          /-
            🎉 no goals
          -/
      right_inv := by
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{u_2, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
          f : Quiver.Hom K L
          x y : Int
          inst✝ : K.HasTotal (ComplexShape.up Int)
          n : Int
          ⊢ Function.RightInverse (fun x_1 => HomologicalComplex₂.instHasTotalIntObjUpSh …
        -/
        intro ⟨⟨a, b⟩, h⟩
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{u_2, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
          f : Quiver.Hom K L
          x y : Int
          inst✝ : K.HasTotal (ComplexShape.up Int)
          n a b : Int
          h : Membership.mem (Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int …
          ⊢ Eq ((fun x_1 => HomologicalComplex₂.instHasTotalIntObjUpShiftFunctor₁.match_ …
        -/
        ext
          /-
            case a.fst
            C : Type u_1
            inst✝² : CategoryTheory.Category.{u_2, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
            f : Quiver.Hom K L
            x y : Int
            inst✝ : K.HasTotal (ComplexShape.up Int)
            n a b : Int
            h : Membership.mem (Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int …
            ⊢ Eq (↑((fun x_1 => HomologicalComplex₂.instHasTotalIntObjUpShiftFunctor₁.matc …
          -/
        · dsimp
          /-
            case a.fst
            C : Type u_1
            inst✝² : CategoryTheory.Category.{u_2, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
            f : Quiver.Hom K L
            x y : Int
            inst✝ : K.HasTotal (ComplexShape.up Int)
            n a b : Int
            h : Membership.mem (Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int …
            ⊢ Eq (HAdd.hAdd (HSub.hSub a x) x) a
          -/
          omega
          /-
            🎉 no goals
          -/
          /-
            case a.snd
            C : Type u_1
            inst✝² : CategoryTheory.Category.{u_2, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
            f : Quiver.Hom K L
            x y : Int
            inst✝ : K.HasTotal (ComplexShape.up Int)
            n a b : Int
            h : Membership.mem (Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int …
            ⊢ Eq (↑((fun x_1 => HomologicalComplex₂.instHasTotalIntObjUpShiftFunctor₁.matc …
          -/
        · rfl }
          /-
            🎉 no goals
          -/
    (fun _ => Iso.refl _)


instance : ((shiftFunctor₂ C y).obj K).HasTotal (up ℤ) := fun n =>
  hasCoproduct_of_equiv_of_iso (K.toGradedObject.mapObjFun (π (up ℤ) (up ℤ) (up ℤ)) (n + y)) _
    { toFun := fun ⟨⟨a, b⟩, h⟩ => ⟨⟨a, b + y⟩, by
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{u_2, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
          f : Quiver.Hom K L
          x y : Int
          inst✝ : K.HasTotal (ComplexShape.up Int)
          n : Int
          x✝ : ↑(Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexSha …
          a b : Int
          h : Membership.mem (Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int …
          ⊢ Membership.mem (Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int)  …
        -/
        simp only [Set.mem_preimage, instTotalComplexShape_π, Set.mem_singleton_iff] at h ⊢
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{u_2, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
          f : Quiver.Hom K L
          x y : Int
          inst✝ : K.HasTotal (ComplexShape.up Int)
          n : Int
          x✝ : ↑(Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexSha …
          a b : Int
          h : Eq (HAdd.hAdd a b) n
          ⊢ Eq (HAdd.hAdd a (HAdd.hAdd b y)) (HAdd.hAdd n y)
        -/
        omega⟩
        /-
          🎉 no goals
        -/
      invFun := fun ⟨⟨a, b⟩, h⟩ => ⟨(a, b - y), by
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{u_2, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
          f : Quiver.Hom K L
          x y : Int
          inst✝ : K.HasTotal (ComplexShape.up Int)
          n : Int
          x✝ : ↑(Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexSha …
          a b : Int
          h : Membership.mem (Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int …
          ⊢ Membership.mem (Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int)  …
        -/
        simp only [Set.mem_preimage, instTotalComplexShape_π, Set.mem_singleton_iff] at h ⊢
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{u_2, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
          f : Quiver.Hom K L
          x y : Int
          inst✝ : K.HasTotal (ComplexShape.up Int)
          n : Int
          x✝ : ↑(Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexSha …
          a b : Int
          h : Eq (HAdd.hAdd a b) (HAdd.hAdd n y)
          ⊢ Eq (HAdd.hAdd a (HSub.hSub b y)) n
        -/
        omega⟩
        /-
          🎉 no goals
        -/
      left_inv := by
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{u_2, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
          f : Quiver.Hom K L
          x y : Int
          inst✝ : K.HasTotal (ComplexShape.up Int)
          n : Int
          ⊢ Function.LeftInverse (fun x => HomologicalComplex₂.instHasTotalIntObjUpShift …
        -/
        rintro ⟨⟨a, b⟩, h⟩
        /-
          case mk.mk
          C : Type u_1
          inst✝² : CategoryTheory.Category.{u_2, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
          f : Quiver.Hom K L
          x y : Int
          inst✝ : K.HasTotal (ComplexShape.up Int)
          n a b : Int
          h : Membership.mem (Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int …
          ⊢ Eq ((fun x => HomologicalComplex₂.instHasTotalIntObjUpShiftFunctor₁.match_2  …
        -/
        ext
          /-
            case mk.mk.a.fst
            C : Type u_1
            inst✝² : CategoryTheory.Category.{u_2, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
            f : Quiver.Hom K L
            x y : Int
            inst✝ : K.HasTotal (ComplexShape.up Int)
            n a b : Int
            h : Membership.mem (Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int …
            ⊢ Eq (↑((fun x => HomologicalComplex₂.instHasTotalIntObjUpShiftFunctor₁.match_ …
          -/
        · rfl
          /-
            🎉 no goals
          -/
          /-
            case mk.mk.a.snd
            C : Type u_1
            inst✝² : CategoryTheory.Category.{u_2, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
            f : Quiver.Hom K L
            x y : Int
            inst✝ : K.HasTotal (ComplexShape.up Int)
            n a b : Int
            h : Membership.mem (Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int …
            ⊢ Eq (↑((fun x => HomologicalComplex₂.instHasTotalIntObjUpShiftFunctor₁.match_ …
          -/
        · dsimp
          /-
            case mk.mk.a.snd
            C : Type u_1
            inst✝² : CategoryTheory.Category.{u_2, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
            f : Quiver.Hom K L
            x y : Int
            inst✝ : K.HasTotal (ComplexShape.up Int)
            n a b : Int
            h : Membership.mem (Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int …
            ⊢ Eq (HSub.hSub (HAdd.hAdd b y) y) b
          -/
          omega
          /-
            🎉 no goals
          -/
      right_inv := by
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{u_2, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
          f : Quiver.Hom K L
          x y : Int
          inst✝ : K.HasTotal (ComplexShape.up Int)
          n : Int
          ⊢ Function.RightInverse (fun x => HomologicalComplex₂.instHasTotalIntObjUpShif …
        -/
        intro ⟨⟨a, b⟩, h⟩
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{u_2, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
          f : Quiver.Hom K L
          x y : Int
          inst✝ : K.HasTotal (ComplexShape.up Int)
          n a b : Int
          h : Membership.mem (Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int …
          ⊢ Eq ((fun x => HomologicalComplex₂.instHasTotalIntObjUpShiftFunctor₁.match_1  …
        -/
        ext
          /-
            case a.fst
            C : Type u_1
            inst✝² : CategoryTheory.Category.{u_2, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
            f : Quiver.Hom K L
            x y : Int
            inst✝ : K.HasTotal (ComplexShape.up Int)
            n a b : Int
            h : Membership.mem (Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int …
            ⊢ Eq (↑((fun x => HomologicalComplex₂.instHasTotalIntObjUpShiftFunctor₁.match_ …
          -/
        · rfl
          /-
            🎉 no goals
          -/
          /-
            case a.snd
            C : Type u_1
            inst✝² : CategoryTheory.Category.{u_2, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
            f : Quiver.Hom K L
            x y : Int
            inst✝ : K.HasTotal (ComplexShape.up Int)
            n a b : Int
            h : Membership.mem (Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int …
            ⊢ Eq (↑((fun x => HomologicalComplex₂.instHasTotalIntObjUpShiftFunctor₁.match_ …
          -/
        · dsimp
          /-
            case a.snd
            C : Type u_1
            inst✝² : CategoryTheory.Category.{u_2, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
            f : Quiver.Hom K L
            x y : Int
            inst✝ : K.HasTotal (ComplexShape.up Int)
            n a b : Int
            h : Membership.mem (Set.preimage ((ComplexShape.up Int).π (ComplexShape.up Int …
            ⊢ Eq (HAdd.hAdd (HSub.hSub b y) y) b
          -/
          omega }
          /-
            🎉 no goals
          -/
    (fun _ => Iso.refl _)


instance : ((shiftFunctor₂ C y ⋙ shiftFunctor₁ C x).obj K).HasTotal (up ℤ) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    f : Quiver.Hom K L
    x y : Int
    inst✝ : K.HasTotal (ComplexShape.up Int)
    ⊢ (((HomologicalComplex₂.shiftFunctor₂ C y).comp (HomologicalComplex₂.shiftFun …
  -/
  dsimp
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    f : Quiver.Hom K L
    x y : Int
    inst✝ : K.HasTotal (ComplexShape.up Int)
    ⊢ ((HomologicalComplex₂.shiftFunctor₁ C x).obj ((HomologicalComplex₂.shiftFunc …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance : ((shiftFunctor₁ C x ⋙ shiftFunctor₂ C y).obj K).HasTotal (up ℤ) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    f : Quiver.Hom K L
    x y : Int
    inst✝ : K.HasTotal (ComplexShape.up Int)
    ⊢ (((HomologicalComplex₂.shiftFunctor₁ C x).comp (HomologicalComplex₂.shiftFun …
  -/
  dsimp
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    f : Quiver.Hom K L
    x y : Int
    inst✝ : K.HasTotal (ComplexShape.up Int)
    ⊢ ((HomologicalComplex₂.shiftFunctor₂ C y).obj ((HomologicalComplex₂.shiftFunc …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- Auxiliary definition for `totalShift₁Iso`. -/
noncomputable def totalShift₁XIso (n n' : ℤ) (h : n + x = n') :
    (((shiftFunctor₁ C x).obj K).total (up ℤ)).X n ≅ (K.total (up ℤ)).X n' where
                                                                      /-
                                                                        C : Type u_1
                                                                        inst✝² : CategoryTheory.Category.{?u.21078, u_1} C
                                                                        inst✝¹ : CategoryTheory.Preadditive C
                                                                        K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
                                                                        f : Quiver.Hom K L
                                                                        x y : Int
                                                                        inst✝ : K.HasTotal (ComplexShape.up Int)
                                                                        n n' : Int
                                                                        h : Eq (HAdd.hAdd n x) n'
                                                                        p q : Int
                                                                        hpq : Eq ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexShape.up Int)  …
                                                                        ⊢ Eq ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexShape.up Int) { fs …
                                                                      -/
  hom := totalDesc _ (fun p q hpq => K.ιTotal (up ℤ) (p + x) q n' (by dsimp at hpq ⊢; omega))
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
  inv := totalDesc _ (fun p q hpq =>
    (K.XXIsoOfEq _ _ _ (Int.sub_add_cancel p x) rfl).inv ≫
      ((shiftFunctor₁ C x).obj K).ιTotal (up ℤ) (p - x) q n
            /-
              C : Type u_1
              inst✝² : CategoryTheory.Category.{?u.21078, u_1} C
              inst✝¹ : CategoryTheory.Preadditive C
              K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
              f : Quiver.Hom K L
              x y : Int
              inst✝ : K.HasTotal (ComplexShape.up Int)
              n n' : Int
              h : Eq (HAdd.hAdd n x) n'
              p q : Int
              hpq : Eq ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexShape.up Int)  …
              ⊢ Eq ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexShape.up Int) { fs …
            -/
        (by dsimp at hpq ⊢; omega))
                            /-
                              🎉 no goals
                            -/
  hom_inv_id := by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.21078, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      f : Quiver.Hom K L
      x y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n n' : Int
      h : Eq (HAdd.hAdd n x) n'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₁  …
    -/
    ext p q h
    /-
      case h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.21078, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      f : Quiver.Hom K L
      x y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n n' : Int
      h✝ : Eq (HAdd.hAdd n x) n'
      p q : Int
      h : Eq ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexShape.up Int) {  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₁  …
    -/
    dsimp
    /-
      case h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.21078, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      f : Quiver.Hom K L
      x y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n n' : Int
      h✝ : Eq (HAdd.hAdd n x) n'
      p q : Int
      h : Eq ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexShape.up Int) {  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₁  …
    -/
    simp only [ι_totalDesc_assoc, CochainComplex.shiftFunctor_obj_X', ι_totalDesc, comp_id]
    /-
      case h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.21078, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      f : Quiver.Hom K L
      x y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n n' : Int
      h✝ : Eq (HAdd.hAdd n x) n'
      p q : Int
      h : Eq ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexShape.up Int) {  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex₂.XXIsoOfEq C (Com …
    -/
    exact ((shiftFunctor₁ C x).obj K).XXIsoOfEq_inv_ιTotal _ (by omega) rfl _ _
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.21078, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      f : Quiver.Hom K L
      x y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n n' : Int
      h : Eq (HAdd.hAdd n x) n'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.totalDesc fun p q hpq => CategoryT …
    -/
    ext
    /-
      case h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.21078, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      f : Quiver.Hom K L
      x y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n n' : Int
      h : Eq (HAdd.hAdd n x) n'
      i₁✝ i₂✝ : Int
      hi✝ : Eq ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexShape.up Int)  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.ιTotal (ComplexShape.up Int) i₁✝ i …
    -/
    dsimp
    /-
      case h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.21078, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      f : Quiver.Hom K L
      x y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n n' : Int
      h : Eq (HAdd.hAdd n x) n'
      i₁✝ i₂✝ : Int
      hi✝ : Eq ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexShape.up Int)  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.ιTotal (ComplexShape.up Int) i₁✝ i …
    -/
    simp only [ι_totalDesc_assoc, Category.assoc, ι_totalDesc, XXIsoOfEq_inv_ιTotal, comp_id]
    /-
      🎉 no goals
    -/


@[reassoc]
lemma D₁_totalShift₁XIso_hom (n₀ n₁ n₀' n₁' : ℤ) (h₀ : n₀ + x = n₀') (h₁ : n₁ + x = n₁') :
    ((shiftFunctor₁ C x).obj K).D₁ (up ℤ) n₀ n₁ ≫ (K.totalShift₁XIso x n₁ n₁' h₁).hom =
      x.negOnePow • ((K.totalShift₁XIso x n₀ n₀' h₀).hom ≫ K.D₁ (up ℤ) n₀' n₁') := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    x : Int
    inst✝ : K.HasTotal (ComplexShape.up Int)
    n₀ n₁ n₀' n₁' : Int
    h₀ : Eq (HAdd.hAdd n₀ x) n₀'
    h₁ : Eq (HAdd.hAdd n₁ x) n₁'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₁  …
  -/
  by_cases h : (up ℤ).Rel n₀ n₁
    /-
      case pos
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      x : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ x) n₀'
      h₁ : Eq (HAdd.hAdd n₁ x) n₁'
      h : (ComplexShape.up Int).Rel n₀ n₁
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₁  …
    -/
  · apply total.hom_ext
    /-
      case pos.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      x : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ x) n₀'
      h₁ : Eq (HAdd.hAdd n₁ x) n₁'
      h : (ComplexShape.up Int).Rel n₀ n₁
      ⊢ ∀ (i₁ i₂ : Int) (hi : Eq ((ComplexShape.up Int).π (ComplexShape.up Int) (Com …
    -/
    intro p q hpq
    /-
      case pos.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      x : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ x) n₀'
      h₁ : Eq (HAdd.hAdd n₁ x) n₁'
      h : (ComplexShape.up Int).Rel n₀ n₁
      p q : Int
      hpq : Eq ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexShape.up Int)  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₁  …
    -/
    dsimp at h hpq
    /-
      case pos.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      x : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ x) n₀'
      h₁ : Eq (HAdd.hAdd n₁ x) n₁'
      h : Eq (HAdd.hAdd n₀ 1) n₁
      p q : Int
      hpq : Eq (HAdd.hAdd p q) n₀
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₁  …
    -/
    dsimp [totalShift₁XIso]
    rw [ι_D₁_assoc, Linear.comp_units_smul, ι_totalDesc_assoc, ι_D₁,
      ((shiftFunctor₁ C x).obj K).d₁_eq _ rfl _ _ (by dsimp; omega),
      K.d₁_eq _ (show p + x + 1 = p + 1 + x by omega) _ _ (by dsimp; omega)]
    /-
      case pos.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      x : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ x) n₀'
      h₁ : Eq (HAdd.hAdd n₁ x) n₁'
      h : Eq (HAdd.hAdd n₀ 1) n₁
      p q : Int
      hpq : Eq (HAdd.hAdd p q) n₀
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSMul.hSMul ((ComplexShape.up Int).ε …
    -/
    dsimp
    /-
      case pos.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      x : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ x) n₀'
      h₁ : Eq (HAdd.hAdd n₁ x) n₁'
      h : Eq (HAdd.hAdd n₀ 1) n₁
      p q : Int
      hpq : Eq (HAdd.hAdd p q) n₀
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSMul.hSMul 1 (CategoryTheory.Catego …
    -/
    rw [one_smul, Category.assoc, ι_totalDesc, one_smul, Linear.units_smul_comp]
    /-
      🎉 no goals
    -/
    /-
      case neg
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      x : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ x) n₀'
      h₁ : Eq (HAdd.hAdd n₁ x) n₁'
      h : Not ((ComplexShape.up Int).Rel n₀ n₁)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₁  …
    -/
  · rw [D₁_shape _ _ _ _ h, zero_comp, D₁_shape, comp_zero, smul_zero]
    /-
      case neg.h₁₂
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      x : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ x) n₀'
      h₁ : Eq (HAdd.hAdd n₁ x) n₁'
      h : Not ((ComplexShape.up Int).Rel n₀ n₁)
      ⊢ Not ((ComplexShape.up Int).Rel n₀' n₁')
    -/
    intro h'
    /-
      case neg.h₁₂
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      x : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ x) n₀'
      h₁ : Eq (HAdd.hAdd n₁ x) n₁'
      h : Not ((ComplexShape.up Int).Rel n₀ n₁)
      h' : (ComplexShape.up Int).Rel n₀' n₁'
      ⊢ False
    -/
    apply h
    /-
      case neg.h₁₂
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      x : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ x) n₀'
      h₁ : Eq (HAdd.hAdd n₁ x) n₁'
      h : Not ((ComplexShape.up Int).Rel n₀ n₁)
      h' : (ComplexShape.up Int).Rel n₀' n₁'
      ⊢ (ComplexShape.up Int).Rel n₀ n₁
    -/
    dsimp at h' ⊢
    /-
      case neg.h₁₂
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      x : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ x) n₀'
      h₁ : Eq (HAdd.hAdd n₁ x) n₁'
      h : Not ((ComplexShape.up Int).Rel n₀ n₁)
      h' : Eq (HAdd.hAdd n₀' 1) n₁'
      ⊢ Eq (HAdd.hAdd n₀ 1) n₁
    -/
    omega
    /-
      🎉 no goals
    -/


@[reassoc]
lemma D₂_totalShift₁XIso_hom (n₀ n₁ n₀' n₁' : ℤ) (h₀ : n₀ + x = n₀') (h₁ : n₁ + x = n₁') :
    ((shiftFunctor₁ C x).obj K).D₂ (up ℤ) n₀ n₁ ≫ (K.totalShift₁XIso x n₁ n₁' h₁).hom =
      x.negOnePow • ((K.totalShift₁XIso x n₀ n₀' h₀).hom ≫ K.D₂ (up ℤ) n₀' n₁') := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    x : Int
    inst✝ : K.HasTotal (ComplexShape.up Int)
    n₀ n₁ n₀' n₁' : Int
    h₀ : Eq (HAdd.hAdd n₀ x) n₀'
    h₁ : Eq (HAdd.hAdd n₁ x) n₁'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₁  …
  -/
  by_cases h : (up ℤ).Rel n₀ n₁
    /-
      case pos
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      x : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ x) n₀'
      h₁ : Eq (HAdd.hAdd n₁ x) n₁'
      h : (ComplexShape.up Int).Rel n₀ n₁
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₁  …
    -/
  · apply total.hom_ext
    /-
      case pos.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      x : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ x) n₀'
      h₁ : Eq (HAdd.hAdd n₁ x) n₁'
      h : (ComplexShape.up Int).Rel n₀ n₁
      ⊢ ∀ (i₁ i₂ : Int) (hi : Eq ((ComplexShape.up Int).π (ComplexShape.up Int) (Com …
    -/
    intro p q hpq
    /-
      case pos.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      x : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ x) n₀'
      h₁ : Eq (HAdd.hAdd n₁ x) n₁'
      h : (ComplexShape.up Int).Rel n₀ n₁
      p q : Int
      hpq : Eq ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexShape.up Int)  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₁  …
    -/
    dsimp at h hpq
    /-
      case pos.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      x : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ x) n₀'
      h₁ : Eq (HAdd.hAdd n₁ x) n₁'
      h : Eq (HAdd.hAdd n₀ 1) n₁
      p q : Int
      hpq : Eq (HAdd.hAdd p q) n₀
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₁  …
    -/
    dsimp [totalShift₁XIso]
    rw [ι_D₂_assoc, Linear.comp_units_smul, ι_totalDesc_assoc, ι_D₂,
      ((shiftFunctor₁ C x).obj K).d₂_eq _ _ rfl _ (by dsimp; omega),
      K.d₂_eq _ _ rfl _ (by dsimp; omega), smul_smul,
      Linear.units_smul_comp, Category.assoc, ι_totalDesc]
    /-
      case pos.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      x : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ x) n₀'
      h₁ : Eq (HAdd.hAdd n₁ x) n₁'
      h : Eq (HAdd.hAdd n₀ 1) n₁
      p q : Int
      hpq : Eq (HAdd.hAdd p q) n₀
      ⊢ Eq (HSMul.hSMul ((ComplexShape.up Int).ε₂ (ComplexShape.up Int) (ComplexShap …
    -/
    dsimp
    /-
      case pos.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      x : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ x) n₀'
      h₁ : Eq (HAdd.hAdd n₁ x) n₁'
      h : Eq (HAdd.hAdd n₀ 1) n₁
      p q : Int
      hpq : Eq (HAdd.hAdd p q) n₀
      ⊢ Eq (HSMul.hSMul p.negOnePow (CategoryTheory.CategoryStruct.comp ((K.X (HAdd. …
    -/
    congr 1
    /-
      case pos.h.e_a
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      x : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ x) n₀'
      h₁ : Eq (HAdd.hAdd n₁ x) n₁'
      h : Eq (HAdd.hAdd n₀ 1) n₁
      p q : Int
      hpq : Eq (HAdd.hAdd p q) n₀
      ⊢ Eq p.negOnePow (HMul.hMul x.negOnePow (HAdd.hAdd p x).negOnePow)
    -/
    rw [add_comm p, Int.negOnePow_add, ← mul_assoc, Int.units_mul_self, one_mul]
    /-
      🎉 no goals
    -/
    /-
      case neg
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      x : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ x) n₀'
      h₁ : Eq (HAdd.hAdd n₁ x) n₁'
      h : Not ((ComplexShape.up Int).Rel n₀ n₁)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₁  …
    -/
  · rw [D₂_shape _ _ _ _ h, zero_comp, D₂_shape, comp_zero, smul_zero]
    /-
      case neg.h₁₂
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      x : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ x) n₀'
      h₁ : Eq (HAdd.hAdd n₁ x) n₁'
      h : Not ((ComplexShape.up Int).Rel n₀ n₁)
      ⊢ Not ((ComplexShape.up Int).Rel n₀' n₁')
    -/
    intro h'
    /-
      case neg.h₁₂
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      x : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ x) n₀'
      h₁ : Eq (HAdd.hAdd n₁ x) n₁'
      h : Not ((ComplexShape.up Int).Rel n₀ n₁)
      h' : (ComplexShape.up Int).Rel n₀' n₁'
      ⊢ False
    -/
    apply h
    /-
      case neg.h₁₂
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      x : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ x) n₀'
      h₁ : Eq (HAdd.hAdd n₁ x) n₁'
      h : Not ((ComplexShape.up Int).Rel n₀ n₁)
      h' : (ComplexShape.up Int).Rel n₀' n₁'
      ⊢ (ComplexShape.up Int).Rel n₀ n₁
    -/
    dsimp at h' ⊢
    /-
      case neg.h₁₂
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      x : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ x) n₀'
      h₁ : Eq (HAdd.hAdd n₁ x) n₁'
      h : Not ((ComplexShape.up Int).Rel n₀ n₁)
      h' : Eq (HAdd.hAdd n₀' 1) n₁'
      ⊢ Eq (HAdd.hAdd n₀ 1) n₁
    -/
    omega
    /-
      🎉 no goals
    -/


/-- The isomorphism `((shiftFunctor₁ C x).obj K).total (up ℤ) ≅ (K.total (up ℤ))⟦x⟧`
expressing the compatibility of the total complex with the shift on the first indices.
This isomorphism does not involve signs. -/
noncomputable def totalShift₁Iso :
    ((shiftFunctor₁ C x).obj K).total (up ℤ) ≅ (K.total (up ℤ))⟦x⟧ :=
  HomologicalComplex.Hom.isoOfComponents (fun n => K.totalShift₁XIso x n (n + x) rfl)
    (fun n n' _ => by
      /-
        C : Type u_1
        inst✝² : CategoryTheory.Category.{?u.85854, u_1} C
        inst✝¹ : CategoryTheory.Preadditive C
        K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
        f : Quiver.Hom K L
        x y : Int
        inst✝ : K.HasTotal (ComplexShape.up Int)
        n n' : Int
        x✝ : (ComplexShape.up Int).Rel n n'
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun n => K.totalShift₁XIso x n (HAd …
      -/
      dsimp
      simp only [total_d, Preadditive.add_comp, Preadditive.comp_add, smul_add,
        Linear.comp_units_smul, K.D₁_totalShift₁XIso_hom x n n' _ _ rfl rfl,
        K.D₂_totalShift₁XIso_hom x n n' _ _ rfl rfl])


@[reassoc]
lemma ι_totalShift₁Iso_hom_f (a b n : ℤ) (h : a + b = n) (a' : ℤ) (ha' : a' = a + x)
    (n' : ℤ) (hn' : n' = n + x) :
    ((shiftFunctor₁ C x).obj K).ιTotal (up ℤ) a b n h ≫ (K.totalShift₁Iso x).hom.f n =
                                                                            /-
                                                                              C : Type u_1
                                                                              inst✝² : CategoryTheory.Category.{?u.93175, u_1} C
                                                                              inst✝¹ : CategoryTheory.Preadditive C
                                                                              K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
                                                                              f : Quiver.Hom K L
                                                                              x y : Int
                                                                              inst✝ : K.HasTotal (ComplexShape.up Int)
                                                                              a b n : Int
                                                                              h : Eq (HAdd.hAdd a b) n
                                                                              a' : Int
                                                                              ha' : Eq a' (HAdd.hAdd a x)
                                                                              n' : Int
                                                                              hn' : Eq n' (HAdd.hAdd n x)
                                                                              ⊢ Eq ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexShape.up Int) { fs …
                                                                            -/
      (K.shiftFunctor₁XXIso a x a' ha' b).hom ≫ K.ιTotal (up ℤ) a' b n' (by dsimp; omega) ≫
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
        (CochainComplex.shiftFunctorObjXIso (K.total (up ℤ)) x n n' hn').inv := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    x : Int
    inst✝ : K.HasTotal (ComplexShape.up Int)
    a b n : Int
    h : Eq (HAdd.hAdd a b) n
    a' : Int
    ha' : Eq a' (HAdd.hAdd a x)
    n' : Int
    hn' : Eq n' (HAdd.hAdd n x)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₁  …
  -/
  subst ha' hn'
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    x : Int
    inst✝ : K.HasTotal (ComplexShape.up Int)
    a b n : Int
    h : Eq (HAdd.hAdd a b) n
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₁  …
  -/
  dsimp [totalShift₁Iso, totalShift₁XIso]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    x : Int
    inst✝ : K.HasTotal (ComplexShape.up Int)
    a b n : Int
    h : Eq (HAdd.hAdd a b) n
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₁  …
  -/
  simp only [ι_totalDesc, comp_id, id_comp]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma ι_totalShift₁Iso_inv_f (a b n : ℤ) (h : a + b = n) (a' n' : ℤ)
    (ha' : a' + b = n') (hn' : n' = n + x) :
    K.ιTotal (up ℤ) a' b n' ha' ≫
      (CochainComplex.shiftFunctorObjXIso (K.total (up ℤ)) x n n' hn').inv ≫
        (K.totalShift₁Iso x).inv.f n =
                                       /-
                                         C : Type u_1
                                         inst✝² : CategoryTheory.Category.{?u.98332, u_1} C
                                         inst✝¹ : CategoryTheory.Preadditive C
                                         K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
                                         f : Quiver.Hom K L
                                         x y : Int
                                         inst✝ : K.HasTotal (ComplexShape.up Int)
                                         a b n : Int
                                         h : Eq (HAdd.hAdd a b) n
                                         a' n' : Int
                                         ha' : Eq (HAdd.hAdd a' b) n'
                                         hn' : Eq n' (HAdd.hAdd n x)
                                         ⊢ Eq a' (HAdd.hAdd a x)
                                       -/
      (K.shiftFunctor₁XXIso a x a' (by omega) b).inv ≫
                                       /-
                                         🎉 no goals
                                       -/
        ((shiftFunctor₁ C x).obj K).ιTotal (up ℤ) a b n h := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    x : Int
    inst✝ : K.HasTotal (ComplexShape.up Int)
    a b n : Int
    h : Eq (HAdd.hAdd a b) n
    a' n' : Int
    ha' : Eq (HAdd.hAdd a' b) n'
    hn' : Eq n' (HAdd.hAdd n x)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.ιTotal (ComplexShape.up Int) a' b  …
  -/
  subst hn'
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    x : Int
    inst✝ : K.HasTotal (ComplexShape.up Int)
    a b n : Int
    h : Eq (HAdd.hAdd a b) n
    a' : Int
    ha' : Eq (HAdd.hAdd a' b) (HAdd.hAdd n x)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.ιTotal (ComplexShape.up Int) a' b  …
  -/
  obtain rfl : a = a' - x := by omega
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    x : Int
    inst✝ : K.HasTotal (ComplexShape.up Int)
    b n a' : Int
    ha' : Eq (HAdd.hAdd a' b) (HAdd.hAdd n x)
    h : Eq (HAdd.hAdd (HSub.hSub a' x) b) n
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.ιTotal (ComplexShape.up Int) a' b  …
  -/
  dsimp [totalShift₁Iso, totalShift₁XIso, shiftFunctor₁XXIso, XXIsoOfEq]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    x : Int
    inst✝ : K.HasTotal (ComplexShape.up Int)
    b n a' : Int
    ha' : Eq (HAdd.hAdd a' b) (HAdd.hAdd n x)
    h : Eq (HAdd.hAdd (HSub.hSub a' x) b) n
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.ιTotal (ComplexShape.up Int) a' b  …
  -/
  simp only [id_comp, ι_totalDesc]
  /-
    🎉 no goals
  -/


variable {K L} in
@[reassoc]
lemma totalShift₁Iso_hom_naturality [L.HasTotal (up ℤ)] :
    total.map ((shiftFunctor₁ C x).map f) (up ℤ) ≫ (L.totalShift₁Iso x).hom =
      (K.totalShift₁Iso x).hom ≫ (total.map f (up ℤ))⟦x⟧' := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    f : Quiver.Hom K L
    x : Int
    inst✝¹ : K.HasTotal (ComplexShape.up Int)
    inst✝ : L.HasTotal (ComplexShape.up Int)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex₂.total.map ((Homo …
  -/
  ext n i₁ i₂ h
  /-
    case h.h
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    f : Quiver.Hom K L
    x : Int
    inst✝¹ : K.HasTotal (ComplexShape.up Int)
    inst✝ : L.HasTotal (ComplexShape.up Int)
    n i₁ i₂ : Int
    h : Eq ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexShape.up Int) {  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₁  …
  -/
  dsimp at h ⊢
  rw [ιTotal_map_assoc, L.ι_totalShift₁Iso_hom_f x i₁ i₂ n h _ rfl _ rfl,
    K.ι_totalShift₁Iso_hom_f_assoc x i₁ i₂ n h _ rfl _ rfl]
  /-
    case h.h
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    f : Quiver.Hom K L
    x : Int
    inst✝¹ : K.HasTotal (ComplexShape.up Int)
    inst✝ : L.HasTotal (ComplexShape.up Int)
    n i₁ i₂ : Int
    h : Eq (HAdd.hAdd i₁ i₂) n
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((HomologicalComplex₂.shiftFunctor₁ …
  -/
  dsimp
  /-
    case h.h
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    f : Quiver.Hom K L
    x : Int
    inst✝¹ : K.HasTotal (ComplexShape.up Int)
    inst✝ : L.HasTotal (ComplexShape.up Int)
    n i₁ i₂ : Int
    h : Eq (HAdd.hAdd i₁ i₂) n
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((f.f (HAdd.hAdd i₁ x)).f i₂) (Catego …
  -/
  rw [id_comp, id_comp, id_comp, comp_id, ιTotal_map]
  /-
    🎉 no goals
  -/


/-- Auxiliary definition for `totalShift₂Iso`. -/
noncomputable def totalShift₂XIso (n n' : ℤ) (h : n + y = n') :
    (((shiftFunctor₂ C y).obj K).total (up ℤ)).X n ≅ (K.total (up ℤ)).X n' where
  hom := totalDesc _ (fun p q hpq => (p * y).negOnePow • K.ιTotal (up ℤ) p (q + y) n'
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.128468, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
          f : Quiver.Hom K L
          x y : Int
          inst✝ : K.HasTotal (ComplexShape.up Int)
          n n' : Int
          h : Eq (HAdd.hAdd n y) n'
          p q : Int
          hpq : Eq ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexShape.up Int)  …
          ⊢ Eq ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexShape.up Int) { fs …
        -/
    (by dsimp at hpq ⊢; omega))
                        /-
                          🎉 no goals
                        -/
  inv := totalDesc _ (fun p q hpq => (p * y).negOnePow •
    (K.XXIsoOfEq _ _ _ rfl (Int.sub_add_cancel q y)).inv ≫
                                                                /-
                                                                  C : Type u_1
                                                                  inst✝² : CategoryTheory.Category.{?u.128468, u_1} C
                                                                  inst✝¹ : CategoryTheory.Preadditive C
                                                                  K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
                                                                  f : Quiver.Hom K L
                                                                  x y : Int
                                                                  inst✝ : K.HasTotal (ComplexShape.up Int)
                                                                  n n' : Int
                                                                  h : Eq (HAdd.hAdd n y) n'
                                                                  p q : Int
                                                                  hpq : Eq ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexShape.up Int)  …
                                                                  ⊢ Eq ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexShape.up Int) { fs …
                                                                -/
      ((shiftFunctor₂ C y).obj K).ιTotal (up ℤ) p (q - y) n (by dsimp at hpq ⊢; omega))
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
  hom_inv_id := by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.128468, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      f : Quiver.Hom K L
      x y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n n' : Int
      h : Eq (HAdd.hAdd n y) n'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₂  …
    -/
    ext p q h
    /-
      case h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.128468, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      f : Quiver.Hom K L
      x y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n n' : Int
      h✝ : Eq (HAdd.hAdd n y) n'
      p q : Int
      h : Eq ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexShape.up Int) {  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₂  …
    -/
    dsimp
    simp only [ι_totalDesc_assoc, Linear.units_smul_comp, ι_totalDesc, smul_smul,
      Int.units_mul_self, one_smul, comp_id]
    /-
      case h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.128468, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      f : Quiver.Hom K L
      x y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n n' : Int
      h✝ : Eq (HAdd.hAdd n y) n'
      p q : Int
      h : Eq ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexShape.up Int) {  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex₂.XXIsoOfEq C (Com …
    -/
    exact ((shiftFunctor₂ C y).obj K).XXIsoOfEq_inv_ιTotal _ rfl (by omega) _ _
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.128468, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      f : Quiver.Hom K L
      x y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n n' : Int
      h : Eq (HAdd.hAdd n y) n'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.totalDesc fun p q hpq => HSMul.hSM …
    -/
    ext
    /-
      case h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.128468, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      f : Quiver.Hom K L
      x y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n n' : Int
      h : Eq (HAdd.hAdd n y) n'
      i₁✝ i₂✝ : Int
      hi✝ : Eq ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexShape.up Int)  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.ιTotal (ComplexShape.up Int) i₁✝ i …
    -/
    dsimp
    simp only [ι_totalDesc_assoc, Linear.units_smul_comp, Category.assoc, ι_totalDesc,
      Linear.comp_units_smul, XXIsoOfEq_inv_ιTotal, smul_smul, Int.units_mul_self, one_smul,
      comp_id]


@[reassoc]
lemma D₁_totalShift₂XIso_hom (n₀ n₁ n₀' n₁' : ℤ) (h₀ : n₀ + y = n₀') (h₁ : n₁ + y = n₁') :
    ((shiftFunctor₂ C y).obj K).D₁ (up ℤ) n₀ n₁ ≫ (K.totalShift₂XIso y n₁ n₁' h₁).hom =
      y.negOnePow • ((K.totalShift₂XIso y n₀ n₀' h₀).hom ≫ K.D₁ (up ℤ) n₀' n₁') := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    y : Int
    inst✝ : K.HasTotal (ComplexShape.up Int)
    n₀ n₁ n₀' n₁' : Int
    h₀ : Eq (HAdd.hAdd n₀ y) n₀'
    h₁ : Eq (HAdd.hAdd n₁ y) n₁'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₂  …
  -/
  by_cases h : (up ℤ).Rel n₀ n₁
    /-
      case pos
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ y) n₀'
      h₁ : Eq (HAdd.hAdd n₁ y) n₁'
      h : (ComplexShape.up Int).Rel n₀ n₁
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₂  …
    -/
  · apply total.hom_ext
    /-
      case pos.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ y) n₀'
      h₁ : Eq (HAdd.hAdd n₁ y) n₁'
      h : (ComplexShape.up Int).Rel n₀ n₁
      ⊢ ∀ (i₁ i₂ : Int) (hi : Eq ((ComplexShape.up Int).π (ComplexShape.up Int) (Com …
    -/
    intro p q hpq
    /-
      case pos.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ y) n₀'
      h₁ : Eq (HAdd.hAdd n₁ y) n₁'
      h : (ComplexShape.up Int).Rel n₀ n₁
      p q : Int
      hpq : Eq ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexShape.up Int)  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₂  …
    -/
    dsimp at h hpq
    /-
      case pos.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ y) n₀'
      h₁ : Eq (HAdd.hAdd n₁ y) n₁'
      h : Eq (HAdd.hAdd n₀ 1) n₁
      p q : Int
      hpq : Eq (HAdd.hAdd p q) n₀
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₂  …
    -/
    dsimp [totalShift₂XIso]
    rw [ι_D₁_assoc, Linear.comp_units_smul, ι_totalDesc_assoc, Linear.units_smul_comp,
      ι_D₁, smul_smul, ((shiftFunctor₂ C y).obj K).d₁_eq _ rfl _ _ (by dsimp; omega),
      K.d₁_eq _ rfl _ _ (by dsimp; omega)]
    /-
      case pos.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ y) n₀'
      h₁ : Eq (HAdd.hAdd n₁ y) n₁'
      h : Eq (HAdd.hAdd n₀ 1) n₁
      p q : Int
      hpq : Eq (HAdd.hAdd p q) n₀
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSMul.hSMul ((ComplexShape.up Int).ε …
    -/
    dsimp
    rw [one_smul, one_smul, Category.assoc, ι_totalDesc, Linear.comp_units_smul,
      ← Int.negOnePow_add]
    /-
      case pos.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ y) n₀'
      h₁ : Eq (HAdd.hAdd n₁ y) n₁'
      h : Eq (HAdd.hAdd n₀ 1) n₁
      p q : Int
      hpq : Eq (HAdd.hAdd p q) n₀
      ⊢ Eq (HSMul.hSMul (HMul.hMul (HAdd.hAdd p 1) y).negOnePow (CategoryTheory.Cate …
    -/
    congr 2
    /-
      case pos.h.e_a.e_n
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ y) n₀'
      h₁ : Eq (HAdd.hAdd n₁ y) n₁'
      h : Eq (HAdd.hAdd n₀ 1) n₁
      p q : Int
      hpq : Eq (HAdd.hAdd p q) n₀
      ⊢ Eq (HMul.hMul (HAdd.hAdd p 1) y) (HAdd.hAdd y (HMul.hMul p y))
    -/
    linarith
    /-
      🎉 no goals
    -/
    /-
      case neg
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ y) n₀'
      h₁ : Eq (HAdd.hAdd n₁ y) n₁'
      h : Not ((ComplexShape.up Int).Rel n₀ n₁)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₂  …
    -/
  · rw [D₁_shape _ _ _ _ h, zero_comp, D₁_shape, comp_zero, smul_zero]
    /-
      case neg.h₁₂
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ y) n₀'
      h₁ : Eq (HAdd.hAdd n₁ y) n₁'
      h : Not ((ComplexShape.up Int).Rel n₀ n₁)
      ⊢ Not ((ComplexShape.up Int).Rel n₀' n₁')
    -/
    intro h'
    /-
      case neg.h₁₂
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ y) n₀'
      h₁ : Eq (HAdd.hAdd n₁ y) n₁'
      h : Not ((ComplexShape.up Int).Rel n₀ n₁)
      h' : (ComplexShape.up Int).Rel n₀' n₁'
      ⊢ False
    -/
    apply h
    /-
      case neg.h₁₂
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ y) n₀'
      h₁ : Eq (HAdd.hAdd n₁ y) n₁'
      h : Not ((ComplexShape.up Int).Rel n₀ n₁)
      h' : (ComplexShape.up Int).Rel n₀' n₁'
      ⊢ (ComplexShape.up Int).Rel n₀ n₁
    -/
    dsimp at h' ⊢
    /-
      case neg.h₁₂
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ y) n₀'
      h₁ : Eq (HAdd.hAdd n₁ y) n₁'
      h : Not ((ComplexShape.up Int).Rel n₀ n₁)
      h' : Eq (HAdd.hAdd n₀' 1) n₁'
      ⊢ Eq (HAdd.hAdd n₀ 1) n₁
    -/
    omega
    /-
      🎉 no goals
    -/


@[reassoc]
lemma D₂_totalShift₂XIso_hom (n₀ n₁ n₀' n₁' : ℤ) (h₀ : n₀ + y = n₀') (h₁ : n₁ + y = n₁') :
    ((shiftFunctor₂ C y).obj K).D₂ (up ℤ) n₀ n₁ ≫ (K.totalShift₂XIso y n₁ n₁' h₁).hom =
      y.negOnePow • ((K.totalShift₂XIso y n₀ n₀' h₀).hom ≫ K.D₂ (up ℤ) n₀' n₁') := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    y : Int
    inst✝ : K.HasTotal (ComplexShape.up Int)
    n₀ n₁ n₀' n₁' : Int
    h₀ : Eq (HAdd.hAdd n₀ y) n₀'
    h₁ : Eq (HAdd.hAdd n₁ y) n₁'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₂  …
  -/
  by_cases h : (up ℤ).Rel n₀ n₁
    /-
      case pos
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ y) n₀'
      h₁ : Eq (HAdd.hAdd n₁ y) n₁'
      h : (ComplexShape.up Int).Rel n₀ n₁
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₂  …
    -/
  · apply total.hom_ext
    /-
      case pos.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ y) n₀'
      h₁ : Eq (HAdd.hAdd n₁ y) n₁'
      h : (ComplexShape.up Int).Rel n₀ n₁
      ⊢ ∀ (i₁ i₂ : Int) (hi : Eq ((ComplexShape.up Int).π (ComplexShape.up Int) (Com …
    -/
    intro p q hpq
    /-
      case pos.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ y) n₀'
      h₁ : Eq (HAdd.hAdd n₁ y) n₁'
      h : (ComplexShape.up Int).Rel n₀ n₁
      p q : Int
      hpq : Eq ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexShape.up Int)  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₂  …
    -/
    dsimp at h hpq
    /-
      case pos.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ y) n₀'
      h₁ : Eq (HAdd.hAdd n₁ y) n₁'
      h : Eq (HAdd.hAdd n₀ 1) n₁
      p q : Int
      hpq : Eq (HAdd.hAdd p q) n₀
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₂  …
    -/
    dsimp [totalShift₂XIso]
    rw [ι_D₂_assoc, Linear.comp_units_smul, ι_totalDesc_assoc, Linear.units_smul_comp,
      smul_smul, ι_D₂, ((shiftFunctor₂ C y).obj K).d₂_eq _ _ rfl _ (by dsimp; omega),
      K.d₂_eq _ _ (show q + y + 1 = q + 1 + y by omega) _ (by dsimp; omega),
      Linear.units_smul_comp, Category.assoc, smul_smul, ι_totalDesc]
    /-
      case pos.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ y) n₀'
      h₁ : Eq (HAdd.hAdd n₁ y) n₁'
      h : Eq (HAdd.hAdd n₀ 1) n₁
      p q : Int
      hpq : Eq (HAdd.hAdd p q) n₀
      ⊢ Eq (HSMul.hSMul ((ComplexShape.up Int).ε₂ (ComplexShape.up Int) (ComplexShap …
    -/
    dsimp
    rw [Linear.units_smul_comp, Linear.comp_units_smul, smul_smul, smul_smul,
      ← Int.negOnePow_add, ← Int.negOnePow_add, ← Int.negOnePow_add,
      ← Int.negOnePow_add]
    /-
      case pos.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ y) n₀'
      h₁ : Eq (HAdd.hAdd n₁ y) n₁'
      h : Eq (HAdd.hAdd n₀ 1) n₁
      p q : Int
      hpq : Eq (HAdd.hAdd p q) n₀
      ⊢ Eq (HSMul.hSMul (HAdd.hAdd (HAdd.hAdd p y) (HMul.hMul p y)).negOnePow (Categ …
    -/
    congr 2
    /-
      case pos.h.e_a.e_n
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ y) n₀'
      h₁ : Eq (HAdd.hAdd n₁ y) n₁'
      h : Eq (HAdd.hAdd n₀ 1) n₁
      p q : Int
      hpq : Eq (HAdd.hAdd p q) n₀
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd p y) (HMul.hMul p y)) (HAdd.hAdd (HAdd.hAdd y (HMul …
    -/
    omega
    /-
      🎉 no goals
    -/
    /-
      case neg
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ y) n₀'
      h₁ : Eq (HAdd.hAdd n₁ y) n₁'
      h : Not ((ComplexShape.up Int).Rel n₀ n₁)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₂  …
    -/
  · rw [D₂_shape _ _ _ _ h, zero_comp, D₂_shape, comp_zero, smul_zero]
    /-
      case neg.h₁₂
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ y) n₀'
      h₁ : Eq (HAdd.hAdd n₁ y) n₁'
      h : Not ((ComplexShape.up Int).Rel n₀ n₁)
      ⊢ Not ((ComplexShape.up Int).Rel n₀' n₁')
    -/
    intro h'
    /-
      case neg.h₁₂
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ y) n₀'
      h₁ : Eq (HAdd.hAdd n₁ y) n₁'
      h : Not ((ComplexShape.up Int).Rel n₀ n₁)
      h' : (ComplexShape.up Int).Rel n₀' n₁'
      ⊢ False
    -/
    apply h
    /-
      case neg.h₁₂
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ y) n₀'
      h₁ : Eq (HAdd.hAdd n₁ y) n₁'
      h : Not ((ComplexShape.up Int).Rel n₀ n₁)
      h' : (ComplexShape.up Int).Rel n₀' n₁'
      ⊢ (ComplexShape.up Int).Rel n₀ n₁
    -/
    dsimp at h' ⊢
    /-
      case neg.h₁₂
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
      y : Int
      inst✝ : K.HasTotal (ComplexShape.up Int)
      n₀ n₁ n₀' n₁' : Int
      h₀ : Eq (HAdd.hAdd n₀ y) n₀'
      h₁ : Eq (HAdd.hAdd n₁ y) n₁'
      h : Not ((ComplexShape.up Int).Rel n₀ n₁)
      h' : Eq (HAdd.hAdd n₀' 1) n₁'
      ⊢ Eq (HAdd.hAdd n₀ 1) n₁
    -/
    omega
    /-
      🎉 no goals
    -/


/-- The isomorphism `((shiftFunctor₂ C y).obj K).total (up ℤ) ≅ (K.total (up ℤ))⟦y⟧`
expressing the compatibility of the total complex with the shift on the second indices.
This isomorphism involves signs: on the summand in degree `(p, q)` of `K`, it is given by the
multiplication by `(p * y).negOnePow`. -/
noncomputable def totalShift₂Iso :
    ((shiftFunctor₂ C y).obj K).total (up ℤ) ≅ (K.total (up ℤ))⟦y⟧ :=
  HomologicalComplex.Hom.isoOfComponents (fun n => K.totalShift₂XIso y n (n + y) rfl)
    (fun n n' _ => by
      /-
        C : Type u_1
        inst✝² : CategoryTheory.Category.{?u.221932, u_1} C
        inst✝¹ : CategoryTheory.Preadditive C
        K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
        f : Quiver.Hom K L
        x y : Int
        inst✝ : K.HasTotal (ComplexShape.up Int)
        n n' : Int
        x✝ : (ComplexShape.up Int).Rel n n'
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun n => K.totalShift₂XIso y n (HAd …
      -/
      dsimp
      simp only [total_d, Preadditive.add_comp, Preadditive.comp_add, smul_add,
        Linear.comp_units_smul, K.D₁_totalShift₂XIso_hom y n n' _ _ rfl rfl,
        K.D₂_totalShift₂XIso_hom y n n' _ _ rfl rfl])


@[reassoc]
lemma ι_totalShift₂Iso_hom_f (a b n : ℤ) (h : a + b = n) (b' : ℤ) (hb' : b' = b + y)
    (n' : ℤ) (hn' : n' = n + y) :
    ((shiftFunctor₂ C y).obj K).ιTotal (up ℤ) a b n h ≫ (K.totalShift₂Iso y).hom.f n =
      (a * y).negOnePow • (K.shiftFunctor₂XXIso a b y b' hb').hom ≫
                                    /-
                                      C : Type u_1
                                      inst✝² : CategoryTheory.Category.{?u.227897, u_1} C
                                      inst✝¹ : CategoryTheory.Preadditive C
                                      K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
                                      f : Quiver.Hom K L
                                      x y : Int
                                      inst✝ : K.HasTotal (ComplexShape.up Int)
                                      a b n : Int
                                      h : Eq (HAdd.hAdd a b) n
                                      b' : Int
                                      hb' : Eq b' (HAdd.hAdd b y)
                                      n' : Int
                                      hn' : Eq n' (HAdd.hAdd n y)
                                      ⊢ Eq ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexShape.up Int) { fs …
                                    -/
        K.ιTotal (up ℤ) a b' n' (by dsimp; omega) ≫
                                           /-
                                             🎉 no goals
                                           -/
          (CochainComplex.shiftFunctorObjXIso (K.total (up ℤ)) y n n' hn').inv := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    y : Int
    inst✝ : K.HasTotal (ComplexShape.up Int)
    a b n : Int
    h : Eq (HAdd.hAdd a b) n
    b' : Int
    hb' : Eq b' (HAdd.hAdd b y)
    n' : Int
    hn' : Eq n' (HAdd.hAdd n y)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₂  …
  -/
  subst hb' hn'
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    y : Int
    inst✝ : K.HasTotal (ComplexShape.up Int)
    a b n : Int
    h : Eq (HAdd.hAdd a b) n
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₂  …
  -/
  dsimp [totalShift₂Iso, totalShift₂XIso]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    y : Int
    inst✝ : K.HasTotal (ComplexShape.up Int)
    a b n : Int
    h : Eq (HAdd.hAdd a b) n
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₂  …
  -/
  simp only [ι_totalDesc, comp_id, id_comp]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma ι_totalShift₂Iso_inv_f (a b n : ℤ) (h : a + b = n) (b' n' : ℤ)
    (hb' : a + b' = n') (hn' : n' = n + y) :
    K.ιTotal (up ℤ) a b' n' hb' ≫
      (CochainComplex.shiftFunctorObjXIso (K.total (up ℤ)) y n n' hn').inv ≫
        (K.totalShift₂Iso y).inv.f n =
                                                             /-
                                                               C : Type u_1
                                                               inst✝² : CategoryTheory.Category.{?u.232769, u_1} C
                                                               inst✝¹ : CategoryTheory.Preadditive C
                                                               K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
                                                               f : Quiver.Hom K L
                                                               x y : Int
                                                               inst✝ : K.HasTotal (ComplexShape.up Int)
                                                               a b n : Int
                                                               h : Eq (HAdd.hAdd a b) n
                                                               b' n' : Int
                                                               hb' : Eq (HAdd.hAdd a b') n'
                                                               hn' : Eq n' (HAdd.hAdd n y)
                                                               ⊢ Eq b' (HAdd.hAdd b y)
                                                             -/
      (a * y).negOnePow • (K.shiftFunctor₂XXIso a b y b' (by omega)).inv ≫
                                                             /-
                                                               🎉 no goals
                                                             -/
        ((shiftFunctor₂ C y).obj K).ιTotal (up ℤ) a b n h := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    y : Int
    inst✝ : K.HasTotal (ComplexShape.up Int)
    a b n : Int
    h : Eq (HAdd.hAdd a b) n
    b' n' : Int
    hb' : Eq (HAdd.hAdd a b') n'
    hn' : Eq n' (HAdd.hAdd n y)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.ιTotal (ComplexShape.up Int) a b'  …
  -/
  subst hn'
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    y : Int
    inst✝ : K.HasTotal (ComplexShape.up Int)
    a b n : Int
    h : Eq (HAdd.hAdd a b) n
    b' : Int
    hb' : Eq (HAdd.hAdd a b') (HAdd.hAdd n y)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.ιTotal (ComplexShape.up Int) a b'  …
  -/
  obtain rfl : b = b' - y := by omega
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    y : Int
    inst✝ : K.HasTotal (ComplexShape.up Int)
    a n b' : Int
    hb' : Eq (HAdd.hAdd a b') (HAdd.hAdd n y)
    h : Eq (HAdd.hAdd a (HSub.hSub b' y)) n
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.ιTotal (ComplexShape.up Int) a b'  …
  -/
  dsimp [totalShift₂Iso, totalShift₂XIso, shiftFunctor₂XXIso, XXIsoOfEq]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    y : Int
    inst✝ : K.HasTotal (ComplexShape.up Int)
    a n b' : Int
    hb' : Eq (HAdd.hAdd a b') (HAdd.hAdd n y)
    h : Eq (HAdd.hAdd a (HSub.hSub b' y)) n
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.ιTotal (ComplexShape.up Int) a b'  …
  -/
  simp only [id_comp, ι_totalDesc]
  /-
    🎉 no goals
  -/


variable {K L} in
@[reassoc]
lemma totalShift₂Iso_hom_naturality [L.HasTotal (up ℤ)] :
    total.map ((shiftFunctor₂ C y).map f) (up ℤ) ≫ (L.totalShift₂Iso y).hom =
      (K.totalShift₂Iso y).hom ≫ (total.map f (up ℤ))⟦y⟧' := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    f : Quiver.Hom K L
    y : Int
    inst✝¹ : K.HasTotal (ComplexShape.up Int)
    inst✝ : L.HasTotal (ComplexShape.up Int)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex₂.total.map ((Homo …
  -/
  ext n i₁ i₂ h
  /-
    case h.h
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    f : Quiver.Hom K L
    y : Int
    inst✝¹ : K.HasTotal (ComplexShape.up Int)
    inst✝ : L.HasTotal (ComplexShape.up Int)
    n i₁ i₂ : Int
    h : Eq ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexShape.up Int) {  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₂  …
  -/
  dsimp at h ⊢
  rw [ιTotal_map_assoc, L.ι_totalShift₂Iso_hom_f y i₁ i₂ n h _ rfl _ rfl,
    K.ι_totalShift₂Iso_hom_f_assoc y i₁ i₂ n h _ rfl _ rfl]
  /-
    case h.h
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    K L : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    f : Quiver.Hom K L
    y : Int
    inst✝¹ : K.HasTotal (ComplexShape.up Int)
    inst✝ : L.HasTotal (ComplexShape.up Int)
    n i₁ i₂ : Int
    h : Eq (HAdd.hAdd i₁ i₂) n
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((HomologicalComplex₂.shiftFunctor₂ …
  -/
  dsimp
  rw [id_comp, id_comp, comp_id, comp_id, Linear.comp_units_smul,
    Linear.units_smul_comp, ιTotal_map]


variable (C) in
/-- The shift functors `shiftFunctor₁ C x` and `shiftFunctor₂ C y` on bicomplexes
with respect to both variables commute. -/
def shiftFunctor₁₂CommIso (x y : ℤ) :
    shiftFunctor₂ C y ⋙ shiftFunctor₁ C x ≅ shiftFunctor₁ C x ⋙ shiftFunctor₂ C y :=
  Iso.refl _


/-- The compatibility isomorphisms of the total complex with the shifts
in both variables "commute" only up to a sign `(x * y).negOnePow`. -/
lemma totalShift₁Iso_trans_totalShift₂Iso :
    ((shiftFunctor₂ C y).obj K).totalShift₁Iso x ≪≫
      (shiftFunctor (CochainComplex C ℤ) x).mapIso (K.totalShift₂Iso y) =
    (x * y).negOnePow • (total.mapIso ((shiftFunctor₁₂CommIso C x y).app K) (up ℤ)) ≪≫
      ((shiftFunctor₁ C x).obj K).totalShift₂Iso y ≪≫
      (shiftFunctor _ y).mapIso (K.totalShift₁Iso x) ≪≫
      (shiftFunctorComm (CochainComplex C ℤ) x y).app _ := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    x y : Int
    inst✝ : K.HasTotal (ComplexShape.up Int)
    ⊢ Eq ((((HomologicalComplex₂.shiftFunctor₂ C y).obj K).totalShift₁Iso x).trans …
  -/
  ext n n₁ n₂ h
  /-
    case w.h.h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    x y : Int
    inst✝ : K.HasTotal (ComplexShape.up Int)
    n n₁ n₂ : Int
    h : Eq ((ComplexShape.up Int).π (ComplexShape.up Int) (ComplexShape.up Int) {  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₁  …
  -/
  dsimp at h ⊢
  rw [Linear.comp_units_smul,ι_totalShift₁Iso_hom_f_assoc _ x n₁ n₂ n h _ rfl _ rfl,
    ιTotal_map_assoc, ι_totalShift₂Iso_hom_f_assoc _ y n₁ n₂ n h _ rfl _ rfl,
    Linear.units_smul_comp, Linear.comp_units_smul]
  /-
    case w.h.h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    x y : Int
    inst✝ : K.HasTotal (ComplexShape.up Int)
    n n₁ n₂ : Int
    h : Eq (HAdd.hAdd n₁ n₂) n
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex₂.shiftFunctor₂  …
  -/
  dsimp [shiftFunctor₁₂CommIso]
  rw [id_comp, id_comp, id_comp, id_comp, comp_id,
    ι_totalShift₂Iso_hom_f _ y (n₁ + x) n₂ (n + x) (by omega) _ rfl _ rfl, smul_smul,
    ← Int.negOnePow_add, add_mul, add_comm (x * y)]
  /-
    case w.h.h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    x y : Int
    inst✝ : K.HasTotal (ComplexShape.up Int)
    n n₁ n₂ : Int
    h : Eq (HAdd.hAdd n₁ n₂) n
    ⊢ Eq (HSMul.hSMul (HAdd.hAdd (HMul.hMul n₁ y) (HMul.hMul x y)).negOnePow (Cate …
  -/
  dsimp
  rw [id_comp, comp_id,
    ι_totalShift₁Iso_hom_f_assoc _ x n₁ (n₂ + y) (n + y) (by omega) _ rfl (n + x + y) (by omega),
    CochainComplex.shiftFunctorComm_hom_app_f]
  /-
    case w.h.h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    x y : Int
    inst✝ : K.HasTotal (ComplexShape.up Int)
    n n₁ n₂ : Int
    h : Eq (HAdd.hAdd n₁ n₂) n
    ⊢ Eq (HSMul.hSMul (HAdd.hAdd (HMul.hMul n₁ y) (HMul.hMul x y)).negOnePow (K.ιT …
  -/
  dsimp
  /-
    case w.h.h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : HomologicalComplex₂ C (ComplexShape.up Int) (ComplexShape.up Int)
    x y : Int
    inst✝ : K.HasTotal (ComplexShape.up Int)
    n n₁ n₂ : Int
    h : Eq (HAdd.hAdd n₁ n₂) n
    ⊢ Eq (HSMul.hSMul (HAdd.hAdd (HMul.hMul n₁ y) (HMul.hMul x y)).negOnePow (K.ιT …
  -/
  rw [Iso.inv_hom_id, comp_id, id_comp]
  /-
    🎉 no goals
  -/


/-- The compatibility isomorphisms of the total complex with the shifts
in both variables "commute" only up to a sign `(x * y).negOnePow`. -/
@[reassoc]
lemma totalShift₁Iso_hom_totalShift₂Iso_hom :
    (((shiftFunctor₂ C y).obj K).totalShift₁Iso x).hom ≫ (K.totalShift₂Iso y).hom⟦x⟧' =
      (x * y).negOnePow • (total.map ((shiftFunctor₁₂CommIso C x y).hom.app K) (up ℤ) ≫
          (((shiftFunctor₁ C x).obj K).totalShift₂Iso y).hom ≫
          (K.totalShift₁Iso x).hom⟦y⟧' ≫
          (shiftFunctorComm (CochainComplex C ℤ) x y).hom.app _) :=
  congr_arg Iso.hom (totalShift₁Iso_trans_totalShift₂Iso K x y)


