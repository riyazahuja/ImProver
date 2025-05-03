/-- The addition on the morphisms in the category `Quotient r` when `r` is compatible
with the addition. -/
def add (hr : ∀ ⦃X Y : C⦄ (f₁ f₂ g₁ g₂ : X ⟶ Y) (_ : r f₁ f₂) (_ : r g₁ g₂), r (f₁ + g₁) (f₂ + g₂))
    {X Y : Quotient r} (f g : X ⟶ Y) : X ⟶ Y :=
  Quot.liftOn₂ f g (fun a b => Quot.mk _ (a + b))
    (fun f g₁ g₂ h₁₂ => by
      /-
        C : Type ?u.219
        inst✝² : CategoryTheory.Category.{?u.223, ?u.219} C
        inst✝¹ : CategoryTheory.Preadditive C
        r : HomRel C
        inst✝ : CategoryTheory.Congruence r
        hr : ∀ ⦃X Y : C⦄ (f₁ f₂ g₁ g₂ : Quiver.Hom X Y), r f₁ f₂ → r g₁ g₂ → r (HAdd.h …
        X Y : CategoryTheory.Quotient r
        f✝ g : Quiver.Hom X Y
        f g₁ g₂ : Quiver.Hom X.as Y.as
        h₁₂ : CategoryTheory.Quotient.CompClosure r g₁ g₂
        ⊢ Eq ((fun a b => Quot.mk (CategoryTheory.Quotient.CompClosure r) (HAdd.hAdd a …
      -/
      simp only [compClosure_iff_self] at h₁₂
      /-
        C : Type ?u.219
        inst✝² : CategoryTheory.Category.{?u.223, ?u.219} C
        inst✝¹ : CategoryTheory.Preadditive C
        r : HomRel C
        inst✝ : CategoryTheory.Congruence r
        hr : ∀ ⦃X Y : C⦄ (f₁ f₂ g₁ g₂ : Quiver.Hom X Y), r f₁ f₂ → r g₁ g₂ → r (HAdd.h …
        X Y : CategoryTheory.Quotient r
        f✝ g : Quiver.Hom X Y
        f g₁ g₂ : Quiver.Hom X.as Y.as
        h₁₂ : r g₁ g₂
        ⊢ Eq ((fun a b => Quot.mk (CategoryTheory.Quotient.CompClosure r) (HAdd.hAdd a …
      -/
      erw [functor_map_eq_iff]
      /-
        C : Type ?u.219
        inst✝² : CategoryTheory.Category.{?u.223, ?u.219} C
        inst✝¹ : CategoryTheory.Preadditive C
        r : HomRel C
        inst✝ : CategoryTheory.Congruence r
        hr : ∀ ⦃X Y : C⦄ (f₁ f₂ g₁ g₂ : Quiver.Hom X Y), r f₁ f₂ → r g₁ g₂ → r (HAdd.h …
        X Y : CategoryTheory.Quotient r
        f✝ g : Quiver.Hom X Y
        f g₁ g₂ : Quiver.Hom X.as Y.as
        h₁₂ : r g₁ g₂
        ⊢ r (HAdd.hAdd f g₁) (HAdd.hAdd f g₂)
      -/
      exact hr _ _ _ _ (Congruence.equivalence.refl f) h₁₂)
      /-
        🎉 no goals
      -/
    (fun f₁ f₂ g h₁₂ => by
      /-
        C : Type ?u.219
        inst✝² : CategoryTheory.Category.{?u.223, ?u.219} C
        inst✝¹ : CategoryTheory.Preadditive C
        r : HomRel C
        inst✝ : CategoryTheory.Congruence r
        hr : ∀ ⦃X Y : C⦄ (f₁ f₂ g₁ g₂ : Quiver.Hom X Y), r f₁ f₂ → r g₁ g₂ → r (HAdd.h …
        X Y : CategoryTheory.Quotient r
        f g✝ : Quiver.Hom X Y
        f₁ f₂ g : Quiver.Hom X.as Y.as
        h₁₂ : CategoryTheory.Quotient.CompClosure r f₁ f₂
        ⊢ Eq ((fun a b => Quot.mk (CategoryTheory.Quotient.CompClosure r) (HAdd.hAdd a …
      -/
      simp only [compClosure_iff_self] at h₁₂
      /-
        C : Type ?u.219
        inst✝² : CategoryTheory.Category.{?u.223, ?u.219} C
        inst✝¹ : CategoryTheory.Preadditive C
        r : HomRel C
        inst✝ : CategoryTheory.Congruence r
        hr : ∀ ⦃X Y : C⦄ (f₁ f₂ g₁ g₂ : Quiver.Hom X Y), r f₁ f₂ → r g₁ g₂ → r (HAdd.h …
        X Y : CategoryTheory.Quotient r
        f g✝ : Quiver.Hom X Y
        f₁ f₂ g : Quiver.Hom X.as Y.as
        h₁₂ : r f₁ f₂
        ⊢ Eq ((fun a b => Quot.mk (CategoryTheory.Quotient.CompClosure r) (HAdd.hAdd a …
      -/
      erw [functor_map_eq_iff]
      /-
        C : Type ?u.219
        inst✝² : CategoryTheory.Category.{?u.223, ?u.219} C
        inst✝¹ : CategoryTheory.Preadditive C
        r : HomRel C
        inst✝ : CategoryTheory.Congruence r
        hr : ∀ ⦃X Y : C⦄ (f₁ f₂ g₁ g₂ : Quiver.Hom X Y), r f₁ f₂ → r g₁ g₂ → r (HAdd.h …
        X Y : CategoryTheory.Quotient r
        f g✝ : Quiver.Hom X Y
        f₁ f₂ g : Quiver.Hom X.as Y.as
        h₁₂ : r f₁ f₂
        ⊢ r (HAdd.hAdd f₁ g) (HAdd.hAdd f₂ g)
      -/
      exact hr _ _ _ _ h₁₂ (Congruence.equivalence.refl g))
      /-
        🎉 no goals
      -/


/-- The negation on the morphisms in the category `Quotient r` when `r` is compatible
with the addition. -/
def neg (hr : ∀ ⦃X Y : C⦄ (f₁ f₂ g₁ g₂ : X ⟶ Y) (_ : r f₁ f₂) (_ : r g₁ g₂), r (f₁ + g₁) (f₂ + g₂))
    {X Y : Quotient r} (f : X ⟶ Y) : X ⟶ Y :=
  Quot.liftOn f (fun a => Quot.mk _ (-a))
    (fun f g => by
      /-
        C : Type ?u.3366
        inst✝² : CategoryTheory.Category.{?u.3370, ?u.3366} C
        inst✝¹ : CategoryTheory.Preadditive C
        r : HomRel C
        inst✝ : CategoryTheory.Congruence r
        hr : ∀ ⦃X Y : C⦄ (f₁ f₂ g₁ g₂ : Quiver.Hom X Y), r f₁ f₂ → r g₁ g₂ → r (HAdd.h …
        X Y : CategoryTheory.Quotient r
        f✝ : Quiver.Hom X Y
        f g : Quiver.Hom X.as Y.as
        ⊢ CategoryTheory.Quotient.CompClosure r f g → Eq ((fun a => Quot.mk (CategoryT …
      -/
      intro hfg
      /-
        C : Type ?u.3366
        inst✝² : CategoryTheory.Category.{?u.3370, ?u.3366} C
        inst✝¹ : CategoryTheory.Preadditive C
        r : HomRel C
        inst✝ : CategoryTheory.Congruence r
        hr : ∀ ⦃X Y : C⦄ (f₁ f₂ g₁ g₂ : Quiver.Hom X Y), r f₁ f₂ → r g₁ g₂ → r (HAdd.h …
        X Y : CategoryTheory.Quotient r
        f✝ : Quiver.Hom X Y
        f g : Quiver.Hom X.as Y.as
        hfg : CategoryTheory.Quotient.CompClosure r f g
        ⊢ Eq ((fun a => Quot.mk (CategoryTheory.Quotient.CompClosure r) (Neg.neg a)) f …
      -/
      simp only [compClosure_iff_self] at hfg
      /-
        C : Type ?u.3366
        inst✝² : CategoryTheory.Category.{?u.3370, ?u.3366} C
        inst✝¹ : CategoryTheory.Preadditive C
        r : HomRel C
        inst✝ : CategoryTheory.Congruence r
        hr : ∀ ⦃X Y : C⦄ (f₁ f₂ g₁ g₂ : Quiver.Hom X Y), r f₁ f₂ → r g₁ g₂ → r (HAdd.h …
        X Y : CategoryTheory.Quotient r
        f✝ : Quiver.Hom X Y
        f g : Quiver.Hom X.as Y.as
        hfg : r f g
        ⊢ Eq ((fun a => Quot.mk (CategoryTheory.Quotient.CompClosure r) (Neg.neg a)) f …
      -/
      erw [functor_map_eq_iff]
      /-
        C : Type ?u.3366
        inst✝² : CategoryTheory.Category.{?u.3370, ?u.3366} C
        inst✝¹ : CategoryTheory.Preadditive C
        r : HomRel C
        inst✝ : CategoryTheory.Congruence r
        hr : ∀ ⦃X Y : C⦄ (f₁ f₂ g₁ g₂ : Quiver.Hom X Y), r f₁ f₂ → r g₁ g₂ → r (HAdd.h …
        X Y : CategoryTheory.Quotient r
        f✝ : Quiver.Hom X Y
        f g : Quiver.Hom X.as Y.as
        hfg : r f g
        ⊢ r (Neg.neg f) (Neg.neg g)
      -/
      apply Congruence.equivalence.symm
      /-
        C : Type ?u.3366
        inst✝² : CategoryTheory.Category.{?u.3370, ?u.3366} C
        inst✝¹ : CategoryTheory.Preadditive C
        r : HomRel C
        inst✝ : CategoryTheory.Congruence r
        hr : ∀ ⦃X Y : C⦄ (f₁ f₂ g₁ g₂ : Quiver.Hom X Y), r f₁ f₂ → r g₁ g₂ → r (HAdd.h …
        X Y : CategoryTheory.Quotient r
        f✝ : Quiver.Hom X Y
        f g : Quiver.Hom X.as Y.as
        hfg : r f g
        ⊢ r (Neg.neg g) (Neg.neg f)
      -/
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
      convert hr f g _ _ hfg (Congruence.equivalence.refl (-f-g)) using 1 <;> abel)
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


/-- The preadditive structure on the category `Quotient r` when `r` is compatible
with the addition. -/
def preadditive
    (hr : ∀ ⦃X Y : C⦄ (f₁ f₂ g₁ g₂ : X ⟶ Y) (_ : r f₁ f₂) (_ : r g₁ g₂), r (f₁ + g₁) (f₂ + g₂)) :
    Preadditive (Quotient r) where
  homGroup P Q :=
    let iZ : Zero (P ⟶ Q) :=
      { zero := Quot.mk _ 0 }
    let iA : Add (P ⟶ Q) :=
      { add := Preadditive.add r hr }
    let iN : Neg (P ⟶ Q) :=
      { neg := Preadditive.neg r hr }
                      /-
                        C : Type ?u.7245
                        inst✝² : CategoryTheory.Category.{?u.9240, ?u.7245} C
                        inst✝¹ : CategoryTheory.Preadditive C
                        r : HomRel C
                        inst✝ : CategoryTheory.Congruence r
                        hr : ∀ ⦃X Y : C⦄ (f₁ f₂ g₁ g₂ : Quiver.Hom X Y), r f₁ f₂ → r g₁ g₂ → r (HAdd.h …
                        P Q : CategoryTheory.Quotient r
                        iZ : Zero (Quiver.Hom P Q) := { zero := Quot.mk (CategoryTheory.Quotient.CompC …
                        iA : Add (Quiver.Hom P Q) := { add := CategoryTheory.Quotient.Preadditive.add  …
                        iN : Neg (Quiver.Hom P Q) := { neg := CategoryTheory.Quotient.Preadditive.neg  …
                        ⊢ ∀ (a b c : Quiver.Hom P Q), Eq (HAdd.hAdd (HAdd.hAdd a b) c) (HAdd.hAdd a (H …
                      -/
    { add_assoc := by rintro ⟨_⟩ ⟨_⟩ ⟨_⟩; exact congr_arg (functor r).map (add_assoc _ _ _)
                                          /-
                                            🎉 no goals
                                          -/
                     /-
                       C : Type ?u.7245
                       inst✝² : CategoryTheory.Category.{?u.9240, ?u.7245} C
                       inst✝¹ : CategoryTheory.Preadditive C
                       r : HomRel C
                       inst✝ : CategoryTheory.Congruence r
                       hr : ∀ ⦃X Y : C⦄ (f₁ f₂ g₁ g₂ : Quiver.Hom X Y), r f₁ f₂ → r g₁ g₂ → r (HAdd.h …
                       P Q : CategoryTheory.Quotient r
                       iZ : Zero (Quiver.Hom P Q) := { zero := Quot.mk (CategoryTheory.Quotient.CompC …
                       iA : Add (Quiver.Hom P Q) := { add := CategoryTheory.Quotient.Preadditive.add  …
                       iN : Neg (Quiver.Hom P Q) := { neg := CategoryTheory.Quotient.Preadditive.neg  …
                       ⊢ ∀ (a : Quiver.Hom P Q), Eq (HAdd.hAdd 0 a) a
                     -/
      zero_add := by rintro ⟨_⟩; exact congr_arg (functor r).map (zero_add _)
                                 /-
                                   🎉 no goals
                                 -/
                     /-
                       C : Type ?u.7245
                       inst✝² : CategoryTheory.Category.{?u.9240, ?u.7245} C
                       inst✝¹ : CategoryTheory.Preadditive C
                       r : HomRel C
                       inst✝ : CategoryTheory.Congruence r
                       hr : ∀ ⦃X Y : C⦄ (f₁ f₂ g₁ g₂ : Quiver.Hom X Y), r f₁ f₂ → r g₁ g₂ → r (HAdd.h …
                       P Q : CategoryTheory.Quotient r
                       iZ : Zero (Quiver.Hom P Q) := { zero := Quot.mk (CategoryTheory.Quotient.CompC …
                       iA : Add (Quiver.Hom P Q) := { add := CategoryTheory.Quotient.Preadditive.add  …
                       iN : Neg (Quiver.Hom P Q) := { neg := CategoryTheory.Quotient.Preadditive.neg  …
                       ⊢ ∀ (a : Quiver.Hom P Q), Eq (HAdd.hAdd a 0) a
                     -/
      add_zero := by rintro ⟨_⟩; exact congr_arg (functor r).map (add_zero _)
                                 /-
                                   🎉 no goals
                                 -/
                     /-
                       C : Type ?u.7245
                       inst✝² : CategoryTheory.Category.{?u.9240, ?u.7245} C
                       inst✝¹ : CategoryTheory.Preadditive C
                       r : HomRel C
                       inst✝ : CategoryTheory.Congruence r
                       hr : ∀ ⦃X Y : C⦄ (f₁ f₂ g₁ g₂ : Quiver.Hom X Y), r f₁ f₂ → r g₁ g₂ → r (HAdd.h …
                       P Q : CategoryTheory.Quotient r
                       iZ : Zero (Quiver.Hom P Q) := { zero := Quot.mk (CategoryTheory.Quotient.CompC …
                       iA : Add (Quiver.Hom P Q) := { add := CategoryTheory.Quotient.Preadditive.add  …
                       iN : Neg (Quiver.Hom P Q) := { neg := CategoryTheory.Quotient.Preadditive.neg  …
                       ⊢ ∀ (a b : Quiver.Hom P Q), Eq (HAdd.hAdd a b) (HAdd.hAdd b a)
                     -/
                           /-
                             C : Type ?u.7245
                             inst✝² : CategoryTheory.Category.{?u.9240, ?u.7245} C
                             inst✝¹ : CategoryTheory.Preadditive C
                             r : HomRel C
                             inst✝ : CategoryTheory.Congruence r
                             hr : ∀ ⦃X Y : C⦄ (f₁ f₂ g₁ g₂ : Quiver.Hom X Y), r f₁ f₂ → r g₁ g₂ → r (HAdd.h …
                             P Q : CategoryTheory.Quotient r
                             iZ : Zero (Quiver.Hom P Q) := { zero := Quot.mk (CategoryTheory.Quotient.CompC …
                             iA : Add (Quiver.Hom P Q) := { add := CategoryTheory.Quotient.Preadditive.add  …
                             iN : Neg (Quiver.Hom P Q) := { neg := CategoryTheory.Quotient.Preadditive.neg  …
                             ⊢ ∀ (a : Quiver.Hom P Q), Eq (HAdd.hAdd (Neg.neg a) a) 0
                           -/
      add_comm := by rintro ⟨_⟩ ⟨_⟩; exact congr_arg (functor r).map (add_comm _ _)
                                       /-
                                         🎉 no goals
                                       -/
                                     /-
                                       🎉 no goals
                                     -/
      neg_add_cancel := by rintro ⟨_⟩; exact congr_arg (functor r).map (neg_add_cancel _)
      -- todo: use a better defeq
      nsmul := nsmulRec
      zsmul := zsmulRec }
  add_comp := by
    /-
      C : Type ?u.7245
      inst✝² : CategoryTheory.Category.{?u.9240, ?u.7245} C
      inst✝¹ : CategoryTheory.Preadditive C
      r : HomRel C
      inst✝ : CategoryTheory.Congruence r
      hr : ∀ ⦃X Y : C⦄ (f₁ f₂ g₁ g₂ : Quiver.Hom X Y), r f₁ f₂ → r g₁ g₂ → r (HAdd.h …
      ⊢ ∀ (P Q R : CategoryTheory.Quotient r) (f f' : Quiver.Hom P Q) (g : Quiver.Ho …
    -/
    rintro _ _ _ ⟨_⟩ ⟨_⟩ ⟨_⟩
    /-
      case mk.mk.mk
      C : Type ?u.7245
      inst✝² : CategoryTheory.Category.{?u.9240, ?u.7245} C
      inst✝¹ : CategoryTheory.Preadditive C
      r : HomRel C
      inst✝ : CategoryTheory.Congruence r
      hr : ∀ ⦃X Y : C⦄ (f₁ f₂ g₁ g₂ : Quiver.Hom X Y), r f₁ f₂ → r g₁ g₂ → r (HAdd.h …
      P✝ Q✝ R✝ : CategoryTheory.Quotient r
      f✝ : Quiver.Hom P✝ Q✝
      a✝² : Quiver.Hom P✝.as Q✝.as
      f'✝ : Quiver.Hom P✝ Q✝
      a✝¹ : Quiver.Hom P✝.as Q✝.as
      g✝ : Quiver.Hom Q✝ R✝
      a✝ : Quiver.Hom Q✝.as R✝.as
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HAdd.hAdd (Quot.mk (CategoryTheory.Q …
    -/
    exact congr_arg (functor r).map (by apply Preadditive.add_comp)
    /-
      🎉 no goals
    -/
  comp_add := by
    /-
      C : Type ?u.7245
      inst✝² : CategoryTheory.Category.{?u.9240, ?u.7245} C
      inst✝¹ : CategoryTheory.Preadditive C
      r : HomRel C
      inst✝ : CategoryTheory.Congruence r
      hr : ∀ ⦃X Y : C⦄ (f₁ f₂ g₁ g₂ : Quiver.Hom X Y), r f₁ f₂ → r g₁ g₂ → r (HAdd.h …
      ⊢ ∀ (P Q R : CategoryTheory.Quotient r) (f : Quiver.Hom P Q) (g g' : Quiver.Ho …
    -/
    rintro _ _ _ ⟨_⟩ ⟨_⟩ ⟨_⟩
    /-
      case mk.mk.mk
      C : Type ?u.7245
      inst✝² : CategoryTheory.Category.{?u.9240, ?u.7245} C
      inst✝¹ : CategoryTheory.Preadditive C
      r : HomRel C
      inst✝ : CategoryTheory.Congruence r
      hr : ∀ ⦃X Y : C⦄ (f₁ f₂ g₁ g₂ : Quiver.Hom X Y), r f₁ f₂ → r g₁ g₂ → r (HAdd.h …
      P✝ Q✝ R✝ : CategoryTheory.Quotient r
      f✝ : Quiver.Hom P✝ Q✝
      a✝² : Quiver.Hom P✝.as Q✝.as
      g✝ : Quiver.Hom Q✝ R✝
      a✝¹ : Quiver.Hom Q✝.as R✝.as
      g'✝ : Quiver.Hom Q✝ R✝
      a✝ : Quiver.Hom Q✝.as R✝.as
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (Quot.mk (CategoryTheory.Quotient.Com …
    -/
    exact congr_arg (functor r).map (by apply Preadditive.comp_add)
    /-
      🎉 no goals
    -/


lemma functor_additive
    (hr : ∀ ⦃X Y : C⦄ (f₁ f₂ g₁ g₂ : X ⟶ Y) (_ : r f₁ f₂) (_ : r g₁ g₂), r (f₁ + g₁) (f₂ + g₂)) :
    letI := preadditive r hr
    (functor r).Additive :=
  letI := preadditive r hr
  { map_add := rfl }


