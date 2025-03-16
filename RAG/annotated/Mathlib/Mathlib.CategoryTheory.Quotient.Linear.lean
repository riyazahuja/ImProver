/-- The scalar multiplications on morphisms in `Quotient R`. -/
def smul (hr : ∀ (a : R) ⦃X Y : C⦄ (f₁ f₂ : X ⟶ Y) (_ : r f₁ f₂), r (a • f₁) (a • f₂))
    (X Y : Quotient r) : SMul R (X ⟶ Y) where
  smul a := Quot.lift (fun g => Quot.mk _ (a • g)) (fun f₁ f₂ h₁₂ => by
    /-
      R : Type u_1
      C : Type u_2
      inst✝⁴ : Semiring R
      inst✝³ : CategoryTheory.Category.{?u.272, u_2} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Linear R C
      r : HomRel C
      inst✝ : CategoryTheory.Congruence r
      hr : ∀ (a : R) ⦃X Y : C⦄ (f₁ f₂ : Quiver.Hom X Y), r f₁ f₂ → r (HSMul.hSMul a  …
      X Y : CategoryTheory.Quotient r
      a : R
      f₁ f₂ : Quiver.Hom X.as Y.as
      h₁₂ : CategoryTheory.Quotient.CompClosure r f₁ f₂
      ⊢ Eq ((fun g => Quot.mk (CategoryTheory.Quotient.CompClosure r) (HSMul.hSMul a …
    -/
    dsimp
    /-
      R : Type u_1
      C : Type u_2
      inst✝⁴ : Semiring R
      inst✝³ : CategoryTheory.Category.{?u.272, u_2} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Linear R C
      r : HomRel C
      inst✝ : CategoryTheory.Congruence r
      hr : ∀ (a : R) ⦃X Y : C⦄ (f₁ f₂ : Quiver.Hom X Y), r f₁ f₂ → r (HSMul.hSMul a  …
      X Y : CategoryTheory.Quotient r
      a : R
      f₁ f₂ : Quiver.Hom X.as Y.as
      h₁₂ : CategoryTheory.Quotient.CompClosure r f₁ f₂
      ⊢ Eq (Quot.mk (CategoryTheory.Quotient.CompClosure r) (HSMul.hSMul a f₁)) (Quo …
    -/
    simp only [compClosure_eq_self] at h₁₂
    /-
      R : Type u_1
      C : Type u_2
      inst✝⁴ : Semiring R
      inst✝³ : CategoryTheory.Category.{?u.272, u_2} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Linear R C
      r : HomRel C
      inst✝ : CategoryTheory.Congruence r
      hr : ∀ (a : R) ⦃X Y : C⦄ (f₁ f₂ : Quiver.Hom X Y), r f₁ f₂ → r (HSMul.hSMul a  …
      X Y : CategoryTheory.Quotient r
      a : R
      f₁ f₂ : Quiver.Hom X.as Y.as
      h₁₂ : r f₁ f₂
      ⊢ Eq (Quot.mk (CategoryTheory.Quotient.CompClosure r) (HSMul.hSMul a f₁)) (Quo …
    -/
    apply Quot.sound
    /-
      case a
      R : Type u_1
      C : Type u_2
      inst✝⁴ : Semiring R
      inst✝³ : CategoryTheory.Category.{?u.272, u_2} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Linear R C
      r : HomRel C
      inst✝ : CategoryTheory.Congruence r
      hr : ∀ (a : R) ⦃X Y : C⦄ (f₁ f₂ : Quiver.Hom X Y), r f₁ f₂ → r (HSMul.hSMul a  …
      X Y : CategoryTheory.Quotient r
      a : R
      f₁ f₂ : Quiver.Hom X.as Y.as
      h₁₂ : r f₁ f₂
      ⊢ CategoryTheory.Quotient.CompClosure r (HSMul.hSMul a f₁) (HSMul.hSMul a f₂)
    -/
    rw [compClosure_eq_self]
    /-
      case a
      R : Type u_1
      C : Type u_2
      inst✝⁴ : Semiring R
      inst✝³ : CategoryTheory.Category.{?u.272, u_2} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Linear R C
      r : HomRel C
      inst✝ : CategoryTheory.Congruence r
      hr : ∀ (a : R) ⦃X Y : C⦄ (f₁ f₂ : Quiver.Hom X Y), r f₁ f₂ → r (HSMul.hSMul a  …
      X Y : CategoryTheory.Quotient r
      a : R
      f₁ f₂ : Quiver.Hom X.as Y.as
      h₁₂ : r f₁ f₂
      ⊢ r (HSMul.hSMul a f₁) (HSMul.hSMul a f₂)
    -/
    exact hr _ _ _ h₁₂)
    /-
      🎉 no goals
    -/


@[simp]
lemma smul_eq (hr : ∀ (a : R) ⦃X Y : C⦄ (f₁ f₂ : X ⟶ Y) (_ : r f₁ f₂), r (a • f₁) (a • f₂))
    (a : R) {X Y : C} (f : X ⟶ Y) :
    letI := smul r hr
    a • (functor r).map f = (functor r).map (a • f) := rfl



/-- Auxiliary definition for `Quotient.Linear.module`. -/
def module' (hr : ∀ (a : R) ⦃X Y : C⦄ (f₁ f₂ : X ⟶ Y) (_ : r f₁ f₂), r (a • f₁) (a • f₂))
    [Preadditive (Quotient r)] [(functor r).Additive] (X Y : C) :
    Module R ((functor r).obj X ⟶ (functor r).obj Y) :=
  letI smul := smul r hr ((functor r).obj X) ((functor r).obj Y)
  { smul_zero := fun a => by
      /-
        R : Type u_1
        C : Type u_2
        inst✝⁶ : Semiring R
        inst✝⁵ : CategoryTheory.Category.{?u.3388, u_2} C
        inst✝⁴ : CategoryTheory.Preadditive C
        inst✝³ : CategoryTheory.Linear R C
        r : HomRel C
        inst✝² : CategoryTheory.Congruence r
        hr : ∀ (a : R) ⦃X Y : C⦄ (f₁ f₂ : Quiver.Hom X Y), r f₁ f₂ → r (HSMul.hSMul a  …
        inst✝¹ : CategoryTheory.Preadditive (CategoryTheory.Quotient r)
        inst✝ : (CategoryTheory.Quotient.functor r).Additive
        X Y : C
        smul : SMul R (Quiver.Hom ((CategoryTheory.Quotient.functor r).obj X) ((Catego …
        a : R
        ⊢ Eq (HSMul.hSMul a 0) 0
      -/
      dsimp
      /-
        R : Type u_1
        C : Type u_2
        inst✝⁶ : Semiring R
        inst✝⁵ : CategoryTheory.Category.{?u.3388, u_2} C
        inst✝⁴ : CategoryTheory.Preadditive C
        inst✝³ : CategoryTheory.Linear R C
        r : HomRel C
        inst✝² : CategoryTheory.Congruence r
        hr : ∀ (a : R) ⦃X Y : C⦄ (f₁ f₂ : Quiver.Hom X Y), r f₁ f₂ → r (HSMul.hSMul a  …
        inst✝¹ : CategoryTheory.Preadditive (CategoryTheory.Quotient r)
        inst✝ : (CategoryTheory.Quotient.functor r).Additive
        X Y : C
        smul : SMul R (Quiver.Hom ((CategoryTheory.Quotient.functor r).obj X) ((Catego …
        a : R
        ⊢ Eq (HSMul.hSMul a 0) 0
      -/
      rw [← (functor r).map_zero X Y, smul_eq, smul_zero]
      /-
        🎉 no goals
      -/
    zero_smul := fun f => by
      /-
        R : Type u_1
        C : Type u_2
        inst✝⁶ : Semiring R
        inst✝⁵ : CategoryTheory.Category.{?u.3388, u_2} C
        inst✝⁴ : CategoryTheory.Preadditive C
        inst✝³ : CategoryTheory.Linear R C
        r : HomRel C
        inst✝² : CategoryTheory.Congruence r
        hr : ∀ (a : R) ⦃X Y : C⦄ (f₁ f₂ : Quiver.Hom X Y), r f₁ f₂ → r (HSMul.hSMul a  …
        inst✝¹ : CategoryTheory.Preadditive (CategoryTheory.Quotient r)
        inst✝ : (CategoryTheory.Quotient.functor r).Additive
        X Y : C
        smul : SMul R (Quiver.Hom ((CategoryTheory.Quotient.functor r).obj X) ((Catego …
        f : Quiver.Hom ((CategoryTheory.Quotient.functor r).obj X) ((CategoryTheory.Qu …
        ⊢ Eq (HSMul.hSMul 0 f) 0
      -/
      /-
        R : Type u_1
        C : Type u_2
        inst✝⁶ : Semiring R
        inst✝⁵ : CategoryTheory.Category.{?u.3388, u_2} C
        inst✝⁴ : CategoryTheory.Preadditive C
        inst✝³ : CategoryTheory.Linear R C
        r : HomRel C
        inst✝² : CategoryTheory.Congruence r
        hr : ∀ (a : R) ⦃X Y : C⦄ (f₁ f₂ : Quiver.Hom X Y), r f₁ f₂ → r (HSMul.hSMul a  …
        inst✝¹ : CategoryTheory.Preadditive (CategoryTheory.Quotient r)
        inst✝ : (CategoryTheory.Quotient.functor r).Additive
        X Y : C
        smul : SMul R (Quiver.Hom ((CategoryTheory.Quotient.functor r).obj X) ((Catego …
        f : Quiver.Hom ((CategoryTheory.Quotient.functor r).obj X) ((CategoryTheory.Qu …
        ⊢ Eq (HSMul.hSMul 1 f) f
      -/
      obtain ⟨f, rfl⟩ := (functor r).map_surjective f
      /-
        case intro
        R : Type u_1
        C : Type u_2
        inst✝⁶ : Semiring R
        inst✝⁵ : CategoryTheory.Category.{?u.3388, u_2} C
        inst✝⁴ : CategoryTheory.Preadditive C
        inst✝³ : CategoryTheory.Linear R C
        r : HomRel C
        inst✝² : CategoryTheory.Congruence r
        hr : ∀ (a : R) ⦃X Y : C⦄ (f₁ f₂ : Quiver.Hom X Y), r f₁ f₂ → r (HSMul.hSMul a  …
        inst✝¹ : CategoryTheory.Preadditive (CategoryTheory.Quotient r)
        inst✝ : (CategoryTheory.Quotient.functor r).Additive
        X Y : C
        smul : SMul R (Quiver.Hom ((CategoryTheory.Quotient.functor r).obj X) ((Catego …
        f : Quiver.Hom X Y
        ⊢ Eq (HSMul.hSMul 1 ((CategoryTheory.Quotient.functor r).map f)) ((CategoryThe …
      -/
      /-
        case intro
        R : Type u_1
        C : Type u_2
        inst✝⁶ : Semiring R
        inst✝⁵ : CategoryTheory.Category.{?u.3388, u_2} C
        inst✝⁴ : CategoryTheory.Preadditive C
        inst✝³ : CategoryTheory.Linear R C
        r : HomRel C
        inst✝² : CategoryTheory.Congruence r
        hr : ∀ (a : R) ⦃X Y : C⦄ (f₁ f₂ : Quiver.Hom X Y), r f₁ f₂ → r (HSMul.hSMul a  …
        inst✝¹ : CategoryTheory.Preadditive (CategoryTheory.Quotient r)
        inst✝ : (CategoryTheory.Quotient.functor r).Additive
        X Y : C
        smul : SMul R (Quiver.Hom ((CategoryTheory.Quotient.functor r).obj X) ((Catego …
        f : Quiver.Hom X Y
        ⊢ Eq (HSMul.hSMul 0 ((CategoryTheory.Quotient.functor r).map f)) 0
      -/
      /-
        case intro
        R : Type u_1
        C : Type u_2
        inst✝⁶ : Semiring R
        inst✝⁵ : CategoryTheory.Category.{?u.3388, u_2} C
        inst✝⁴ : CategoryTheory.Preadditive C
        inst✝³ : CategoryTheory.Linear R C
        r : HomRel C
        inst✝² : CategoryTheory.Congruence r
        hr : ∀ (a : R) ⦃X Y : C⦄ (f₁ f₂ : Quiver.Hom X Y), r f₁ f₂ → r (HSMul.hSMul a  …
        inst✝¹ : CategoryTheory.Preadditive (CategoryTheory.Quotient r)
        inst✝ : (CategoryTheory.Quotient.functor r).Additive
        X Y : C
        smul : SMul R (Quiver.Hom ((CategoryTheory.Quotient.functor r).obj X) ((Catego …
        f : Quiver.Hom X Y
        ⊢ Eq ((CategoryTheory.Quotient.functor r).map (HSMul.hSMul 1 f)) ((CategoryThe …
      -/
      dsimp [smul]
      /-
        🎉 no goals
      -/
      /-
        case intro
        R : Type u_1
        C : Type u_2
        inst✝⁶ : Semiring R
        inst✝⁵ : CategoryTheory.Category.{?u.3388, u_2} C
        inst✝⁴ : CategoryTheory.Preadditive C
        inst✝³ : CategoryTheory.Linear R C
        r : HomRel C
        inst✝² : CategoryTheory.Congruence r
        hr : ∀ (a : R) ⦃X Y : C⦄ (f₁ f₂ : Quiver.Hom X Y), r f₁ f₂ → r (HSMul.hSMul a  …
        inst✝¹ : CategoryTheory.Preadditive (CategoryTheory.Quotient r)
        inst✝ : (CategoryTheory.Quotient.functor r).Additive
        X Y : C
        smul : SMul R (Quiver.Hom ((CategoryTheory.Quotient.functor r).obj X) ((Catego …
        f : Quiver.Hom X Y
        ⊢ Eq ((CategoryTheory.Quotient.functor r).map (HSMul.hSMul 0 f)) 0
      -/
      /-
        R : Type u_1
        C : Type u_2
        inst✝⁶ : Semiring R
        inst✝⁵ : CategoryTheory.Category.{?u.3388, u_2} C
        inst✝⁴ : CategoryTheory.Preadditive C
        inst✝³ : CategoryTheory.Linear R C
        r : HomRel C
        inst✝² : CategoryTheory.Congruence r
        hr : ∀ (a : R) ⦃X Y : C⦄ (f₁ f₂ : Quiver.Hom X Y), r f₁ f₂ → r (HSMul.hSMul a  …
        inst✝¹ : CategoryTheory.Preadditive (CategoryTheory.Quotient r)
        inst✝ : (CategoryTheory.Quotient.functor r).Additive
        X Y : C
        smul : SMul R (Quiver.Hom ((CategoryTheory.Quotient.functor r).obj X) ((Catego …
        a b : R
        f : Quiver.Hom ((CategoryTheory.Quotient.functor r).obj X) ((CategoryTheory.Qu …
        ⊢ Eq (HSMul.hSMul (HMul.hMul a b) f) (HSMul.hSMul a (HSMul.hSMul b f))
      -/
      rw [zero_smul, Functor.map_zero]
      /-
        case intro
        R : Type u_1
        C : Type u_2
        inst✝⁶ : Semiring R
        inst✝⁵ : CategoryTheory.Category.{?u.3388, u_2} C
        inst✝⁴ : CategoryTheory.Preadditive C
        inst✝³ : CategoryTheory.Linear R C
        r : HomRel C
        inst✝² : CategoryTheory.Congruence r
        hr : ∀ (a : R) ⦃X Y : C⦄ (f₁ f₂ : Quiver.Hom X Y), r f₁ f₂ → r (HSMul.hSMul a  …
        inst✝¹ : CategoryTheory.Preadditive (CategoryTheory.Quotient r)
        inst✝ : (CategoryTheory.Quotient.functor r).Additive
        X Y : C
        smul : SMul R (Quiver.Hom ((CategoryTheory.Quotient.functor r).obj X) ((Catego …
        a b : R
        f : Quiver.Hom X Y
        ⊢ Eq (HSMul.hSMul (HMul.hMul a b) ((CategoryTheory.Quotient.functor r).map f)) …
      -/
      /-
        🎉 no goals
      -/
      /-
        case intro
        R : Type u_1
        C : Type u_2
        inst✝⁶ : Semiring R
        inst✝⁵ : CategoryTheory.Category.{?u.3388, u_2} C
        inst✝⁴ : CategoryTheory.Preadditive C
        inst✝³ : CategoryTheory.Linear R C
        r : HomRel C
        inst✝² : CategoryTheory.Congruence r
        hr : ∀ (a : R) ⦃X Y : C⦄ (f₁ f₂ : Quiver.Hom X Y), r f₁ f₂ → r (HSMul.hSMul a  …
        inst✝¹ : CategoryTheory.Preadditive (CategoryTheory.Quotient r)
        inst✝ : (CategoryTheory.Quotient.functor r).Additive
        X Y : C
        smul : SMul R (Quiver.Hom ((CategoryTheory.Quotient.functor r).obj X) ((Catego …
        a b : R
        f : Quiver.Hom X Y
        ⊢ Eq ((CategoryTheory.Quotient.functor r).map (HSMul.hSMul (HMul.hMul a b) f)) …
      -/
    one_smul := fun f => by
      /-
        🎉 no goals
      -/
      obtain ⟨f, rfl⟩ := (functor r).map_surjective f
      dsimp [smul]
      rw [one_smul]
    mul_smul := fun a b f => by
      /-
        R : Type u_1
        C : Type u_2
        inst✝⁶ : Semiring R
        inst✝⁵ : CategoryTheory.Category.{?u.3388, u_2} C
        inst✝⁴ : CategoryTheory.Preadditive C
        inst✝³ : CategoryTheory.Linear R C
        r : HomRel C
        inst✝² : CategoryTheory.Congruence r
        hr : ∀ (a : R) ⦃X Y : C⦄ (f₁ f₂ : Quiver.Hom X Y), r f₁ f₂ → r (HSMul.hSMul a  …
        inst✝¹ : CategoryTheory.Preadditive (CategoryTheory.Quotient r)
        inst✝ : (CategoryTheory.Quotient.functor r).Additive
        X Y : C
        smul : SMul R (Quiver.Hom ((CategoryTheory.Quotient.functor r).obj X) ((Catego …
        a : R
        f g : Quiver.Hom ((CategoryTheory.Quotient.functor r).obj X) ((CategoryTheory. …
        ⊢ Eq (HSMul.hSMul a (HAdd.hAdd f g)) (HAdd.hAdd (HSMul.hSMul a f) (HSMul.hSMul …
      -/
      obtain ⟨f, rfl⟩ := (functor r).map_surjective f
      /-
        case intro
        R : Type u_1
        C : Type u_2
        inst✝⁶ : Semiring R
        inst✝⁵ : CategoryTheory.Category.{?u.3388, u_2} C
        inst✝⁴ : CategoryTheory.Preadditive C
        inst✝³ : CategoryTheory.Linear R C
        r : HomRel C
        inst✝² : CategoryTheory.Congruence r
        hr : ∀ (a : R) ⦃X Y : C⦄ (f₁ f₂ : Quiver.Hom X Y), r f₁ f₂ → r (HSMul.hSMul a  …
        inst✝¹ : CategoryTheory.Preadditive (CategoryTheory.Quotient r)
        inst✝ : (CategoryTheory.Quotient.functor r).Additive
        X Y : C
        smul : SMul R (Quiver.Hom ((CategoryTheory.Quotient.functor r).obj X) ((Catego …
        a : R
        g : Quiver.Hom ((CategoryTheory.Quotient.functor r).obj X) ((CategoryTheory.Qu …
        f : Quiver.Hom X Y
        ⊢ Eq (HSMul.hSMul a (HAdd.hAdd ((CategoryTheory.Quotient.functor r).map f) g)) …
      -/
      dsimp [smul]
      /-
        case intro.intro
        R : Type u_1
        C : Type u_2
        inst✝⁶ : Semiring R
        inst✝⁵ : CategoryTheory.Category.{?u.3388, u_2} C
        inst✝⁴ : CategoryTheory.Preadditive C
        inst✝³ : CategoryTheory.Linear R C
        r : HomRel C
        inst✝² : CategoryTheory.Congruence r
        hr : ∀ (a : R) ⦃X Y : C⦄ (f₁ f₂ : Quiver.Hom X Y), r f₁ f₂ → r (HSMul.hSMul a  …
        inst✝¹ : CategoryTheory.Preadditive (CategoryTheory.Quotient r)
        inst✝ : (CategoryTheory.Quotient.functor r).Additive
        X Y : C
        smul : SMul R (Quiver.Hom ((CategoryTheory.Quotient.functor r).obj X) ((Catego …
        a : R
        f g : Quiver.Hom X Y
        ⊢ Eq (HSMul.hSMul a (HAdd.hAdd ((CategoryTheory.Quotient.functor r).map f) ((C …
      -/
      rw [mul_smul]
      /-
        case intro.intro
        R : Type u_1
        C : Type u_2
        inst✝⁶ : Semiring R
        inst✝⁵ : CategoryTheory.Category.{?u.3388, u_2} C
        inst✝⁴ : CategoryTheory.Preadditive C
        inst✝³ : CategoryTheory.Linear R C
        r : HomRel C
        inst✝² : CategoryTheory.Congruence r
        hr : ∀ (a : R) ⦃X Y : C⦄ (f₁ f₂ : Quiver.Hom X Y), r f₁ f₂ → r (HSMul.hSMul a  …
        inst✝¹ : CategoryTheory.Preadditive (CategoryTheory.Quotient r)
        inst✝ : (CategoryTheory.Quotient.functor r).Additive
        X Y : C
        smul : SMul R (Quiver.Hom ((CategoryTheory.Quotient.functor r).obj X) ((Catego …
        a : R
        f g : Quiver.Hom X Y
        ⊢ Eq (HSMul.hSMul a (HAdd.hAdd ((CategoryTheory.Quotient.functor r).map f) ((C …
      -/
    smul_add := fun a f g => by
      /-
        🎉 no goals
      -/
      obtain ⟨f, rfl⟩ := (functor r).map_surjective f
      /-
        R : Type u_1
        C : Type u_2
        inst✝⁶ : Semiring R
        inst✝⁵ : CategoryTheory.Category.{?u.3388, u_2} C
        inst✝⁴ : CategoryTheory.Preadditive C
        inst✝³ : CategoryTheory.Linear R C
        r : HomRel C
        inst✝² : CategoryTheory.Congruence r
        hr : ∀ (a : R) ⦃X Y : C⦄ (f₁ f₂ : Quiver.Hom X Y), r f₁ f₂ → r (HSMul.hSMul a  …
        inst✝¹ : CategoryTheory.Preadditive (CategoryTheory.Quotient r)
        inst✝ : (CategoryTheory.Quotient.functor r).Additive
        X Y : C
        smul : SMul R (Quiver.Hom ((CategoryTheory.Quotient.functor r).obj X) ((Catego …
        a b : R
        f : Quiver.Hom ((CategoryTheory.Quotient.functor r).obj X) ((CategoryTheory.Qu …
        ⊢ Eq (HSMul.hSMul (HAdd.hAdd a b) f) (HAdd.hAdd (HSMul.hSMul a f) (HSMul.hSMul …
      -/
      obtain ⟨g, rfl⟩ := (functor r).map_surjective g
      /-
        case intro
        R : Type u_1
        C : Type u_2
        inst✝⁶ : Semiring R
        inst✝⁵ : CategoryTheory.Category.{?u.3388, u_2} C
        inst✝⁴ : CategoryTheory.Preadditive C
        inst✝³ : CategoryTheory.Linear R C
        r : HomRel C
        inst✝² : CategoryTheory.Congruence r
        hr : ∀ (a : R) ⦃X Y : C⦄ (f₁ f₂ : Quiver.Hom X Y), r f₁ f₂ → r (HSMul.hSMul a  …
        inst✝¹ : CategoryTheory.Preadditive (CategoryTheory.Quotient r)
        inst✝ : (CategoryTheory.Quotient.functor r).Additive
        X Y : C
        smul : SMul R (Quiver.Hom ((CategoryTheory.Quotient.functor r).obj X) ((Catego …
        a b : R
        f : Quiver.Hom X Y
        ⊢ Eq (HSMul.hSMul (HAdd.hAdd a b) ((CategoryTheory.Quotient.functor r).map f)) …
      -/
      dsimp [smul]
      /-
        case intro
        R : Type u_1
        C : Type u_2
        inst✝⁶ : Semiring R
        inst✝⁵ : CategoryTheory.Category.{?u.3388, u_2} C
        inst✝⁴ : CategoryTheory.Preadditive C
        inst✝³ : CategoryTheory.Linear R C
        r : HomRel C
        inst✝² : CategoryTheory.Congruence r
        hr : ∀ (a : R) ⦃X Y : C⦄ (f₁ f₂ : Quiver.Hom X Y), r f₁ f₂ → r (HSMul.hSMul a  …
        inst✝¹ : CategoryTheory.Preadditive (CategoryTheory.Quotient r)
        inst✝ : (CategoryTheory.Quotient.functor r).Additive
        X Y : C
        smul : SMul R (Quiver.Hom ((CategoryTheory.Quotient.functor r).obj X) ((Catego …
        a b : R
        f : Quiver.Hom X Y
        ⊢ Eq ((CategoryTheory.Quotient.functor r).map (HSMul.hSMul (HAdd.hAdd a b) f)) …
      -/
      rw [← (functor r).map_add, smul_eq, ← (functor r).map_add, smul_add]
      /-
        🎉 no goals
      -/
    add_smul := fun a b f => by
      obtain ⟨f, rfl⟩ := (functor r).map_surjective f
      dsimp [smul]
      rw [add_smul, Functor.map_add] }


/-- Auxiliary definition for `Quotient.linear`. -/
def module (hr : ∀ (a : R) ⦃X Y : C⦄ (f₁ f₂ : X ⟶ Y) (_ : r f₁ f₂), r (a • f₁) (a • f₂))
    [Preadditive (Quotient r)] [(functor r).Additive] (X Y : Quotient r) :
    Module R (X ⟶ Y) := module' r hr X.as Y.as


/-- Assuming `Quotient r` has already been endowed with a preadditive category structure
such that `functor r : C ⥤ Quotient r` is additive, and that `C` has a `R`-linear category
structure compatible with `r`, this is the induced `R`-linear category structure on
`Quotient r`. -/
def linear (hr : ∀ (a : R) ⦃X Y : C⦄ (f₁ f₂ : X ⟶ Y) (_ : r f₁ f₂), r (a • f₁) (a • f₂))
    [Preadditive (Quotient r)] [(functor r).Additive] : Linear R (Quotient r) := by
  /-
    R : Type u_1
    C : Type u_2
    inst✝⁶ : Semiring R
    inst✝⁵ : CategoryTheory.Category.{?u.32764, u_2} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Linear R C
    r : HomRel C
    inst✝² : CategoryTheory.Congruence r
    hr : ∀ (a : R) ⦃X Y : C⦄ (f₁ f₂ : Quiver.Hom X Y), r f₁ f₂ → r (HSMul.hSMul a  …
    inst✝¹ : CategoryTheory.Preadditive (CategoryTheory.Quotient r)
    inst✝ : (CategoryTheory.Quotient.functor r).Additive
    ⊢ CategoryTheory.Linear R (CategoryTheory.Quotient r)
  -/
  letI := Linear.module r hr
  exact
    { smul_comp := by
        rintro ⟨X⟩ ⟨Y⟩ ⟨Z⟩ a f g
        obtain ⟨f, rfl⟩ := (functor r).map_surjective f
        obtain ⟨g, rfl⟩ := (functor r).map_surjective g
        rw [Linear.smul_eq, ← Functor.map_comp, ← Functor.map_comp,
          Linear.smul_eq, Linear.smul_comp]
      comp_smul := by
        rintro ⟨X⟩ ⟨Y⟩ ⟨Z⟩ f a g
        obtain ⟨f, rfl⟩ := (functor r).map_surjective f
        obtain ⟨g, rfl⟩ := (functor r).map_surjective g
        rw [Linear.smul_eq, ← Functor.map_comp, ← Functor.map_comp,
          Linear.smul_eq, Linear.comp_smul] }


instance linear_functor
    (hr : ∀ (a : R) ⦃X Y : C⦄ (f₁ f₂ : X ⟶ Y) (_ : r f₁ f₂), r (a • f₁) (a • f₂))
    [Preadditive (Quotient r)] [(functor r).Additive] :
    letI := linear R r hr; Functor.Linear R (functor r) := by
  /-
    R : Type u_1
    C : Type u_2
    inst✝⁶ : Semiring R
    inst✝⁵ : CategoryTheory.Category.{u_3, u_2} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Linear R C
    r : HomRel C
    inst✝² : CategoryTheory.Congruence r
    hr : ∀ (a : R) ⦃X Y : C⦄ (f₁ f₂ : Quiver.Hom X Y), r f₁ f₂ → r (HSMul.hSMul a  …
    inst✝¹ : CategoryTheory.Preadditive (CategoryTheory.Quotient r)
    inst✝ : (CategoryTheory.Quotient.functor r).Additive
    ⊢ CategoryTheory.Functor.Linear R (CategoryTheory.Quotient.functor r)
  -/
  letI := linear R r hr; exact { }
                         /-
                           🎉 no goals
                         -/


