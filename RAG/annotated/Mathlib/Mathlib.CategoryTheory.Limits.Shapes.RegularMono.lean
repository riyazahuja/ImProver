/-- A regular monomorphism is a morphism which is the equalizer of some parallel pair. -/
class RegularMono (f : X ⟶ Y) where
  /-- An object in `C` -/
  Z : C -- Porting note: violates naming but what is better?
  /-- A map from the codomain of `f` to `Z` -/
  left : Y ⟶ Z
  /-- Another map from the codomain of `f` to `Z` -/
  right : Y ⟶ Z
  /-- `f` equalizes the two maps -/
  w : f ≫ left = f ≫ right := by aesop_cat
  /-- `f` is the equalizer of the two maps -/
  isLimit : IsLimit (Fork.ofι f w)


attribute [reassoc] RegularMono.w


/-- Every regular monomorphism is a monomorphism. -/
instance (priority := 100) RegularMono.mono (f : X ⟶ Y) [RegularMono f] : Mono f :=
  mono_of_isLimit_fork RegularMono.isLimit


instance equalizerRegular (g h : X ⟶ Y) [HasLimit (parallelPair g h)] :
    RegularMono (equalizer.ι g h) where
  Z := Y
  left := g
  right := h
  w := equalizer.condition g h
  isLimit :=
                                                    /-
                                                      C : Type u₁
                                                      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                      X Y : C
                                                      g h : Quiver.Hom X Y
                                                      inst✝ : CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.parallelPair g h)
                                                      ⊢ ∀ (s : CategoryTheory.Limits.Fork g h), Eq (CategoryTheory.CategoryStruct.co …
                                                    -/
    Fork.IsLimit.mk _ (fun s => limit.lift _ s) (by simp) fun s m w => by
                                                    /-
                                                      🎉 no goals
                                                    -/
      /-
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        X Y : C
        g h : Quiver.Hom X Y
        inst✝ : CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.parallelPair g h)
        s : CategoryTheory.Limits.Fork g h
        m : Quiver.Hom s.pt (CategoryTheory.Limits.Fork.ofι (CategoryTheory.Limits.equ …
        w : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ofι ( …
        ⊢ Eq m ((fun s => CategoryTheory.Limits.limit.lift (CategoryTheory.Limits.para …
      -/
      apply equalizer.hom_ext
      /-
        case h
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        X Y : C
        g h : Quiver.Hom X Y
        inst✝ : CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.parallelPair g h)
        s : CategoryTheory.Limits.Fork g h
        m : Quiver.Hom s.pt (CategoryTheory.Limits.Fork.ofι (CategoryTheory.Limits.equ …
        w : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ofι ( …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.equalizer.ι  …
      -/
      simp [← w]
      /-
        🎉 no goals
      -/


/-- Every split monomorphism is a regular monomorphism. -/
instance (priority := 100) RegularMono.ofIsSplitMono (f : X ⟶ Y) [IsSplitMono f] :
    RegularMono f where
  Z := Y
  left := 𝟙 Y
  right := retraction f ≫ f
  isLimit := isSplitMonoEqualizes f


/-- If `f` is a regular mono, then any map `k : W ⟶ Y` equalizing `RegularMono.left` and
    `RegularMono.right` induces a morphism `l : W ⟶ X` such that `l ≫ f = k`. -/
def RegularMono.lift' {W : C} (f : X ⟶ Y) [RegularMono f] (k : W ⟶ Y)
    (h : k ≫ (RegularMono.left : Y ⟶ @RegularMono.Z _ _ _ _ f _) = k ≫ RegularMono.right) :
    { l : W ⟶ X // l ≫ f = k } :=
  Fork.IsLimit.lift' RegularMono.isLimit _ h


/-- The second leg of a pullback cone is a regular monomorphism if the right component is too.

See also `Pullback.sndOfMono` for the basic monomorphism version, and
`regularOfIsPullbackFstOfRegular` for the flipped version.
-/
def regularOfIsPullbackSndOfRegular {P Q R S : C} {f : P ⟶ Q} {g : P ⟶ R} {h : Q ⟶ S} {k : R ⟶ S}
    [hr : RegularMono h] (comm : f ≫ h = g ≫ k) (t : IsLimit (PullbackCone.mk _ _ comm)) :
    RegularMono g where
  Z := hr.Z
  left := k ≫ hr.left
  right := k ≫ hr.right
  w := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y P Q R S : C
      f : Quiver.Hom P Q
      g : Quiver.Hom P R
      h : Quiver.Hom Q S
      k : Quiver.Hom R S
      hr : CategoryTheory.RegularMono h
      comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
      t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk f g c …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.CategoryStruct.comp …
    -/
    repeat (rw [← Category.assoc, ← eq_whisker comm])
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y P Q R S : C
      f : Quiver.Hom P Q
      g : Quiver.Hom P R
      h : Quiver.Hom Q S
      k : Quiver.Hom R S
      hr : CategoryTheory.RegularMono h
      comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
      t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk f g c …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
    -/
    simp only [Category.assoc, hr.w]
    /-
      🎉 no goals
    -/
  isLimit := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y P Q R S : C
      f : Quiver.Hom P Q
      g : Quiver.Hom P R
      h : Quiver.Hom Q S
      k : Quiver.Hom R S
      hr : CategoryTheory.RegularMono h
      comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
      t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk f g c …
      ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι g ⋯)
    -/
    apply Fork.IsLimit.mk' _ _
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y P Q R S : C
      f : Quiver.Hom P Q
      g : Quiver.Hom P R
      h : Quiver.Hom Q S
      k : Quiver.Hom R S
      hr : CategoryTheory.RegularMono h
      comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
      t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk f g c …
      ⊢ (s : CategoryTheory.Limits.Fork (CategoryTheory.CategoryStruct.comp k Catego …
    -/
    intro s
    have l₁ : (Fork.ι s ≫ k) ≫ RegularMono.left = (Fork.ι s ≫ k) ≫ hr.right := by
      rw [Category.assoc, s.condition, Category.assoc]
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y P Q R S : C
      f : Quiver.Hom P Q
      g : Quiver.Hom P R
      h : Quiver.Hom Q S
      k : Quiver.Hom R S
      hr : CategoryTheory.RegularMono h
      comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
      t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk f g c …
      s : CategoryTheory.Limits.Fork (CategoryTheory.CategoryStruct.comp k CategoryT …
      l₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
      ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheo …
    -/
    obtain ⟨l, hl⟩ := Fork.IsLimit.lift' hr.isLimit _ l₁
    /-
      case mk
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y P Q R S : C
      f : Quiver.Hom P Q
      g : Quiver.Hom P R
      h : Quiver.Hom Q S
      k : Quiver.Hom R S
      hr : CategoryTheory.RegularMono h
      comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
      t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk f g c …
      s : CategoryTheory.Limits.Fork (CategoryTheory.CategoryStruct.comp k CategoryT …
      l₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
      l : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
      hl : Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheory.Limits.Fork.ofι  …
      ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheo …
    -/
    obtain ⟨p, _, hp₂⟩ := PullbackCone.IsLimit.lift' t _ _ hl
    /-
      case mk.mk.intro
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y P Q R S : C
      f : Quiver.Hom P Q
      g : Quiver.Hom P R
      h : Quiver.Hom Q S
      k : Quiver.Hom R S
      hr : CategoryTheory.RegularMono h
      comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
      t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk f g c …
      s : CategoryTheory.Limits.Fork (CategoryTheory.CategoryStruct.comp k CategoryT …
      l₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
      l : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
      hl : Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheory.Limits.Fork.ofι  …
      p : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
      left✝ : Eq (CategoryTheory.CategoryStruct.comp p (CategoryTheory.Limits.Pullba …
      hp₂ : Eq (CategoryTheory.CategoryStruct.comp p (CategoryTheory.Limits.Pullback …
      ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheo …
    -/
    refine ⟨p, hp₂, ?_⟩
    /-
      case mk.mk.intro
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y P Q R S : C
      f : Quiver.Hom P Q
      g : Quiver.Hom P R
      h : Quiver.Hom Q S
      k : Quiver.Hom R S
      hr : CategoryTheory.RegularMono h
      comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
      t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk f g c …
      s : CategoryTheory.Limits.Fork (CategoryTheory.CategoryStruct.comp k CategoryT …
      l₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
      l : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
      hl : Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheory.Limits.Fork.ofι  …
      p : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
      left✝ : Eq (CategoryTheory.CategoryStruct.comp p (CategoryTheory.Limits.Pullba …
      hp₂ : Eq (CategoryTheory.CategoryStruct.comp p (CategoryTheory.Limits.Pullback …
      ⊢ ∀ {m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.Walk …
    -/
    intro m w
    /-
      case mk.mk.intro
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y P Q R S : C
      f : Quiver.Hom P Q
      g : Quiver.Hom P R
      h : Quiver.Hom Q S
      k : Quiver.Hom R S
      hr : CategoryTheory.RegularMono h
      comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
      t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk f g c …
      s : CategoryTheory.Limits.Fork (CategoryTheory.CategoryStruct.comp k CategoryT …
      l₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
      l : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
      hl : Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheory.Limits.Fork.ofι  …
      p : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
      left✝ : Eq (CategoryTheory.CategoryStruct.comp p (CategoryTheory.Limits.Pullba …
      hp₂ : Eq (CategoryTheory.CategoryStruct.comp p (CategoryTheory.Limits.Pullback …
      m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
      w : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ofι g …
      ⊢ Eq m p
    -/
    have z : m ≫ g = p ≫ g := w.trans hp₂.symm
    /-
      case mk.mk.intro
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y P Q R S : C
      f : Quiver.Hom P Q
      g : Quiver.Hom P R
      h : Quiver.Hom Q S
      k : Quiver.Hom R S
      hr : CategoryTheory.RegularMono h
      comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
      t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk f g c …
      s : CategoryTheory.Limits.Fork (CategoryTheory.CategoryStruct.comp k CategoryT …
      l₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
      l : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
      hl : Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheory.Limits.Fork.ofι  …
      p : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
      left✝ : Eq (CategoryTheory.CategoryStruct.comp p (CategoryTheory.Limits.Pullba …
      hp₂ : Eq (CategoryTheory.CategoryStruct.comp p (CategoryTheory.Limits.Pullback …
      m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
      w : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ofι g …
      z : Eq (CategoryTheory.CategoryStruct.comp m g) (CategoryTheory.CategoryStruct …
      ⊢ Eq m p
    -/
    apply t.hom_ext
    /-
      case mk.mk.intro
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y P Q R S : C
      f : Quiver.Hom P Q
      g : Quiver.Hom P R
      h : Quiver.Hom Q S
      k : Quiver.Hom R S
      hr : CategoryTheory.RegularMono h
      comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
      t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk f g c …
      s : CategoryTheory.Limits.Fork (CategoryTheory.CategoryStruct.comp k CategoryT …
      l₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
      l : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
      hl : Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheory.Limits.Fork.ofι  …
      p : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
      left✝ : Eq (CategoryTheory.CategoryStruct.comp p (CategoryTheory.Limits.Pullba …
      hp₂ : Eq (CategoryTheory.CategoryStruct.comp p (CategoryTheory.Limits.Pullback …
      m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
      w : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ofι g …
      z : Eq (CategoryTheory.CategoryStruct.comp m g) (CategoryTheory.CategoryStruct …
      ⊢ ∀ (j : CategoryTheory.Limits.WalkingCospan), Eq (CategoryTheory.CategoryStru …
    -/
    apply (PullbackCone.mk f g comm).equalizer_ext
      /-
        case mk.mk.intro.h₀
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        X Y P Q R S : C
        f : Quiver.Hom P Q
        g : Quiver.Hom P R
        h : Quiver.Hom Q S
        k : Quiver.Hom R S
        hr : CategoryTheory.RegularMono h
        comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
        t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk f g c …
        s : CategoryTheory.Limits.Fork (CategoryTheory.CategoryStruct.comp k CategoryT …
        l₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
        l : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
        hl : Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheory.Limits.Fork.ofι  …
        p : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
        left✝ : Eq (CategoryTheory.CategoryStruct.comp p (CategoryTheory.Limits.Pullba …
        hp₂ : Eq (CategoryTheory.CategoryStruct.comp p (CategoryTheory.Limits.Pullback …
        m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
        w : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ofι g …
        z : Eq (CategoryTheory.CategoryStruct.comp m g) (CategoryTheory.CategoryStruct …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.PullbackCone …
      -/
    · erw [← cancel_mono h, Category.assoc, Category.assoc, comm]
      /-
        case mk.mk.intro.h₀
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        X Y P Q R S : C
        f : Quiver.Hom P Q
        g : Quiver.Hom P R
        h : Quiver.Hom Q S
        k : Quiver.Hom R S
        hr : CategoryTheory.RegularMono h
        comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
        t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk f g c …
        s : CategoryTheory.Limits.Fork (CategoryTheory.CategoryStruct.comp k CategoryT …
        l₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
        l : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
        hl : Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheory.Limits.Fork.ofι  …
        p : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
        left✝ : Eq (CategoryTheory.CategoryStruct.comp p (CategoryTheory.Limits.Pullba …
        hp₂ : Eq (CategoryTheory.CategoryStruct.comp p (CategoryTheory.Limits.Pullback …
        m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
        w : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ofι g …
        z : Eq (CategoryTheory.CategoryStruct.comp m g) (CategoryTheory.CategoryStruct …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.comp …
      -/
      simp only [← Category.assoc, eq_whisker z]
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.intro.h₁
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        X Y P Q R S : C
        f : Quiver.Hom P Q
        g : Quiver.Hom P R
        h : Quiver.Hom Q S
        k : Quiver.Hom R S
        hr : CategoryTheory.RegularMono h
        comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
        t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk f g c …
        s : CategoryTheory.Limits.Fork (CategoryTheory.CategoryStruct.comp k CategoryT …
        l₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
        l : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
        hl : Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheory.Limits.Fork.ofι  …
        p : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
        left✝ : Eq (CategoryTheory.CategoryStruct.comp p (CategoryTheory.Limits.Pullba …
        hp₂ : Eq (CategoryTheory.CategoryStruct.comp p (CategoryTheory.Limits.Pullback …
        m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
        w : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ofι g …
        z : Eq (CategoryTheory.CategoryStruct.comp m g) (CategoryTheory.CategoryStruct …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.PullbackCone …
      -/
    · exact z
      /-
        🎉 no goals
      -/


/-- The first leg of a pullback cone is a regular monomorphism if the left component is too.

See also `Pullback.fstOfMono` for the basic monomorphism version, and
`regularOfIsPullbackSndOfRegular` for the flipped version.
-/
def regularOfIsPullbackFstOfRegular {P Q R S : C} {f : P ⟶ Q} {g : P ⟶ R} {h : Q ⟶ S} {k : R ⟶ S}
    [RegularMono k] (comm : f ≫ h = g ≫ k) (t : IsLimit (PullbackCone.mk _ _ comm)) :
    RegularMono f :=
  regularOfIsPullbackSndOfRegular comm.symm (PullbackCone.flipIsLimit t)


instance (priority := 100) strongMono_of_regularMono (f : X ⟶ Y) [RegularMono f] : StrongMono f :=
  StrongMono.mk' (by
      /-
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        X Y : C
        f : Quiver.Hom X Y
        inst✝ : CategoryTheory.RegularMono f
        ⊢ ∀ (X_1 Y_1 : C) (z : Quiver.Hom X_1 Y_1), CategoryTheory.Epi z → ∀ (u : Quiv …
      -/
      intro A B z hz u v sq
      have : v ≫ (RegularMono.left : Y ⟶ RegularMono.Z f) = v ≫ RegularMono.right := by
        apply (cancel_epi z).1
        repeat (rw [← Category.assoc, ← eq_whisker sq.w])
        simp only [Category.assoc, RegularMono.w]
      /-
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        X Y : C
        f : Quiver.Hom X Y
        inst✝ : CategoryTheory.RegularMono f
        A B : C
        z : Quiver.Hom A B
        hz : CategoryTheory.Epi z
        u : Quiver.Hom A X
        v : Quiver.Hom B Y
        sq : CategoryTheory.CommSq u z f v
        this : Eq (CategoryTheory.CategoryStruct.comp v CategoryTheory.RegularMono.lef …
        ⊢ sq.HasLift
      -/
      obtain ⟨t, ht⟩ := RegularMono.lift' _ _ this
      /-
        case mk
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        X Y : C
        f : Quiver.Hom X Y
        inst✝ : CategoryTheory.RegularMono f
        A B : C
        z : Quiver.Hom A B
        hz : CategoryTheory.Epi z
        u : Quiver.Hom A X
        v : Quiver.Hom B Y
        sq : CategoryTheory.CommSq u z f v
        this : Eq (CategoryTheory.CategoryStruct.comp v CategoryTheory.RegularMono.lef …
        t : Quiver.Hom B X
        ht : Eq (CategoryTheory.CategoryStruct.comp t f) v
        ⊢ sq.HasLift
      -/
      refine CommSq.HasLift.mk' ⟨t, (cancel_mono f).1 ?_, ht⟩
      /-
        case mk
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        X Y : C
        f : Quiver.Hom X Y
        inst✝ : CategoryTheory.RegularMono f
        A B : C
        z : Quiver.Hom A B
        hz : CategoryTheory.Epi z
        u : Quiver.Hom A X
        v : Quiver.Hom B Y
        sq : CategoryTheory.CommSq u z f v
        this : Eq (CategoryTheory.CategoryStruct.comp v CategoryTheory.RegularMono.lef …
        t : Quiver.Hom B X
        ht : Eq (CategoryTheory.CategoryStruct.comp t f) v
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp z …
      -/
      simp only [Arrow.mk_hom, Arrow.homMk'_left, Category.assoc, ht, sq.w])
      /-
        🎉 no goals
      -/


/-- A regular monomorphism is an isomorphism if it is an epimorphism. -/
theorem isIso_of_regularMono_of_epi (f : X ⟶ Y) [RegularMono f] [Epi f] : IsIso f :=
  isIso_of_epi_of_strongMono _


/-- A regular mono category is a category in which every monomorphism is regular. -/
class RegularMonoCategory where
  /-- Every monomorphism is a regular monomorphism -/
  regularMonoOfMono : ∀ {X Y : C} (f : X ⟶ Y) [Mono f], RegularMono f


/-- In a category in which every monomorphism is regular, we can express every monomorphism as
    an equalizer. This is not an instance because it would create an instance loop. -/
def regularMonoOfMono [RegularMonoCategory C] (f : X ⟶ Y) [Mono f] : RegularMono f :=
  RegularMonoCategory.regularMonoOfMono _


instance (priority := 100) regularMonoCategoryOfSplitMonoCategory [SplitMonoCategory C] :
    RegularMonoCategory C where
  regularMonoOfMono f _ := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y : C
      inst✝ : CategoryTheory.SplitMonoCategory C
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      x✝ : CategoryTheory.Mono f
      ⊢ CategoryTheory.RegularMono f
    -/
    haveI := isSplitMono_of_mono f
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y : C
      inst✝ : CategoryTheory.SplitMonoCategory C
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      x✝ : CategoryTheory.Mono f
      this : CategoryTheory.IsSplitMono f
      ⊢ CategoryTheory.RegularMono f
    -/
    infer_instance
    /-
      🎉 no goals
    -/


instance (priority := 100) strongMonoCategory_of_regularMonoCategory [RegularMonoCategory C] :
    StrongMonoCategory C where
  strongMono_of_mono f _ := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y : C
      inst✝ : CategoryTheory.RegularMonoCategory C
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      x✝ : CategoryTheory.Mono f
      ⊢ CategoryTheory.StrongMono f
    -/
    haveI := regularMonoOfMono f
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y : C
      inst✝ : CategoryTheory.RegularMonoCategory C
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      x✝ : CategoryTheory.Mono f
      this : CategoryTheory.RegularMono f
      ⊢ CategoryTheory.StrongMono f
    -/
    infer_instance
    /-
      🎉 no goals
    -/


/-- A regular epimorphism is a morphism which is the coequalizer of some parallel pair. -/
class RegularEpi (f : X ⟶ Y) where
  /-- An object from `C` -/
  W : C -- Porting note: violates naming convention but what is better?
  /-- Two maps to the domain of `f` -/
  (left right : W ⟶ X)
  /-- `f` coequalizes the two maps -/
  w : left ≫ f = right ≫ f := by aesop_cat
  /-- `f` is the coequalizer -/
  isColimit : IsColimit (Cofork.ofπ f w)


attribute [reassoc] RegularEpi.w


/-- Every regular epimorphism is an epimorphism. -/
instance (priority := 100) RegularEpi.epi (f : X ⟶ Y) [RegularEpi f] : Epi f :=
  epi_of_isColimit_cofork RegularEpi.isColimit


instance coequalizerRegular (g h : X ⟶ Y) [HasColimit (parallelPair g h)] :
    RegularEpi (coequalizer.π g h) where
  W := X
  left := g
  right := h
  w := coequalizer.condition g h
  isColimit :=
                                                          /-
                                                            C : Type u₁
                                                            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                            X Y : C
                                                            g h : Quiver.Hom X Y
                                                            inst✝ : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.parallelPair g …
                                                            ⊢ ∀ (s : CategoryTheory.Limits.Cofork g h), Eq (CategoryTheory.CategoryStruct. …
                                                          -/
    Cofork.IsColimit.mk _ (fun s => colimit.desc _ s) (by simp) fun s m w => by
                                                          /-
                                                            🎉 no goals
                                                          -/
      /-
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        X Y : C
        g h : Quiver.Hom X Y
        inst✝ : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.parallelPair g …
        s : CategoryTheory.Limits.Cofork g h
        m : Quiver.Hom (CategoryTheory.Limits.Cofork.ofπ (CategoryTheory.Limits.coequa …
        w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ ( …
        ⊢ Eq m ((fun s => CategoryTheory.Limits.colimit.desc (CategoryTheory.Limits.pa …
      -/
      apply coequalizer.hom_ext
      /-
        case h
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        X Y : C
        g h : Quiver.Hom X Y
        inst✝ : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.parallelPair g …
        s : CategoryTheory.Limits.Cofork g h
        m : Quiver.Hom (CategoryTheory.Limits.Cofork.ofπ (CategoryTheory.Limits.coequa …
        w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ ( …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
      -/
      simp [← w]
      /-
        🎉 no goals
      -/


/-- A morphism which is a coequalizer for its kernel pair is a regular epi. -/
noncomputable def regularEpiOfKernelPair {B X : C} (f : X ⟶ B) [HasPullback f f]
    (hc : IsColimit (Cofork.ofπ f pullback.condition)) : RegularEpi f where
  W := pullback f f
  left := pullback.fst f f
  right := pullback.snd f f
  w := pullback.condition
  isColimit := hc


/-- Every split epimorphism is a regular epimorphism. -/
instance (priority := 100) RegularEpi.ofSplitEpi (f : X ⟶ Y) [IsSplitEpi f] : RegularEpi f where
  W := X
  left := 𝟙 X
  right := f ≫ section_ f
  isColimit := isSplitEpiCoequalizes f


/-- If `f` is a regular epi, then every morphism `k : X ⟶ W` coequalizing `RegularEpi.left` and
    `RegularEpi.right` induces `l : Y ⟶ W` such that `f ≫ l = k`. -/
def RegularEpi.desc' {W : C} (f : X ⟶ Y) [RegularEpi f] (k : X ⟶ W)
    (h : (RegularEpi.left : RegularEpi.W f ⟶ X) ≫ k = RegularEpi.right ≫ k) :
    { l : Y ⟶ W // f ≫ l = k } :=
  Cofork.IsColimit.desc' RegularEpi.isColimit _ h


/-- The second leg of a pushout cocone is a regular epimorphism if the right component is too.

See also `Pushout.sndOfEpi` for the basic epimorphism version, and
`regularOfIsPushoutFstOfRegular` for the flipped version.
-/
def regularOfIsPushoutSndOfRegular {P Q R S : C} {f : P ⟶ Q} {g : P ⟶ R} {h : Q ⟶ S} {k : R ⟶ S}
    [gr : RegularEpi g] (comm : f ≫ h = g ≫ k) (t : IsColimit (PushoutCocone.mk _ _ comm)) :
    RegularEpi h where
  W := gr.W
  left := gr.left ≫ f
  right := gr.right ≫ f
          /-
            C : Type u₁
            inst✝ : CategoryTheory.Category.{v₁, u₁} C
            X Y P Q R S : C
            f : Quiver.Hom P Q
            g : Quiver.Hom P R
            h : Quiver.Hom Q S
            k : Quiver.Hom R S
            gr : CategoryTheory.RegularEpi g
            comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
            t : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk h  …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp C …
          -/
  w := by rw [Category.assoc, Category.assoc, comm]; simp only [← Category.assoc, eq_whisker gr.w]
                                                     /-
                                                       🎉 no goals
                                                     -/
  isColimit := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y P Q R S : C
      f : Quiver.Hom P Q
      g : Quiver.Hom P R
      h : Quiver.Hom Q S
      k : Quiver.Hom R S
      gr : CategoryTheory.RegularEpi g
      comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
      t : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk h  …
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofork.ofπ h ⋯)
    -/
    apply Cofork.IsColimit.mk' _ _
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y P Q R S : C
      f : Quiver.Hom P Q
      g : Quiver.Hom P R
      h : Quiver.Hom Q S
      k : Quiver.Hom R S
      gr : CategoryTheory.RegularEpi g
      comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
      t : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk h  …
      ⊢ (s : CategoryTheory.Limits.Cofork (CategoryTheory.CategoryStruct.comp Catego …
    -/
    intro s
    have l₁ : gr.left ≫ f ≫ s.π = gr.right ≫ f ≫ s.π := by
      rw [← Category.assoc, ← Category.assoc, s.condition]
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y P Q R S : C
      f : Quiver.Hom P Q
      g : Quiver.Hom P R
      h : Quiver.Hom Q S
      k : Quiver.Hom R S
      gr : CategoryTheory.RegularEpi g
      comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
      t : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk h  …
      s : CategoryTheory.Limits.Cofork (CategoryTheory.CategoryStruct.comp CategoryT …
      l₁ : Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.RegularEpi.left (Ca …
      ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory …
    -/
    obtain ⟨l, hl⟩ := Cofork.IsColimit.desc' gr.isColimit (f ≫ Cofork.π s) l₁
    /-
      case mk
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y P Q R S : C
      f : Quiver.Hom P Q
      g : Quiver.Hom P R
      h : Quiver.Hom Q S
      k : Quiver.Hom R S
      gr : CategoryTheory.RegularEpi g
      comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
      t : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk h  …
      s : CategoryTheory.Limits.Cofork (CategoryTheory.CategoryStruct.comp CategoryT …
      l₁ : Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.RegularEpi.left (Ca …
      l : Quiver.Hom (CategoryTheory.Limits.Cofork.ofπ g ⋯).pt (((CategoryTheory.Fun …
      hl : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ  …
      ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory …
    -/
    obtain ⟨p, hp₁, _⟩ := PushoutCocone.IsColimit.desc' t _ _ hl.symm
    /-
      case mk.mk.intro
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y P Q R S : C
      f : Quiver.Hom P Q
      g : Quiver.Hom P R
      h : Quiver.Hom Q S
      k : Quiver.Hom R S
      gr : CategoryTheory.RegularEpi g
      comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
      t : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk h  …
      s : CategoryTheory.Limits.Cofork (CategoryTheory.CategoryStruct.comp CategoryT …
      l₁ : Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.RegularEpi.left (Ca …
      l : Quiver.Hom (CategoryTheory.Limits.Cofork.ofπ g ⋯).pt (((CategoryTheory.Fun …
      hl : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ  …
      p : Quiver.Hom (CategoryTheory.Limits.PushoutCocone.mk h k comm).pt (((Categor …
      hp₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCoc …
      right✝ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Pushout …
      ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory …
    -/
    refine ⟨p, hp₁, ?_⟩
    /-
      case mk.mk.intro
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y P Q R S : C
      f : Quiver.Hom P Q
      g : Quiver.Hom P R
      h : Quiver.Hom Q S
      k : Quiver.Hom R S
      gr : CategoryTheory.RegularEpi g
      comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
      t : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk h  …
      s : CategoryTheory.Limits.Cofork (CategoryTheory.CategoryStruct.comp CategoryT …
      l₁ : Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.RegularEpi.left (Ca …
      l : Quiver.Hom (CategoryTheory.Limits.Cofork.ofπ g ⋯).pt (((CategoryTheory.Fun …
      hl : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ  …
      p : Quiver.Hom (CategoryTheory.Limits.PushoutCocone.mk h k comm).pt (((Categor …
      hp₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCoc …
      right✝ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Pushout …
      ⊢ ∀ {m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.Walk …
    -/
    intro m w
    /-
      case mk.mk.intro
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y P Q R S : C
      f : Quiver.Hom P Q
      g : Quiver.Hom P R
      h : Quiver.Hom Q S
      k : Quiver.Hom R S
      gr : CategoryTheory.RegularEpi g
      comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
      t : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk h  …
      s : CategoryTheory.Limits.Cofork (CategoryTheory.CategoryStruct.comp CategoryT …
      l₁ : Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.RegularEpi.left (Ca …
      l : Quiver.Hom (CategoryTheory.Limits.Cofork.ofπ g ⋯).pt (((CategoryTheory.Fun …
      hl : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ  …
      p : Quiver.Hom (CategoryTheory.Limits.PushoutCocone.mk h k comm).pt (((Categor …
      hp₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCoc …
      right✝ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Pushout …
      m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
      w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ h …
      ⊢ Eq m p
    -/
    have z := w.trans hp₁.symm
    /-
      case mk.mk.intro
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y P Q R S : C
      f : Quiver.Hom P Q
      g : Quiver.Hom P R
      h : Quiver.Hom Q S
      k : Quiver.Hom R S
      gr : CategoryTheory.RegularEpi g
      comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
      t : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk h  …
      s : CategoryTheory.Limits.Cofork (CategoryTheory.CategoryStruct.comp CategoryT …
      l₁ : Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.RegularEpi.left (Ca …
      l : Quiver.Hom (CategoryTheory.Limits.Cofork.ofπ g ⋯).pt (((CategoryTheory.Fun …
      hl : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ  …
      p : Quiver.Hom (CategoryTheory.Limits.PushoutCocone.mk h k comm).pt (((Categor …
      hp₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCoc …
      right✝ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Pushout …
      m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
      w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ h …
      z : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ h …
      ⊢ Eq m p
    -/
    apply t.hom_ext
    /-
      case mk.mk.intro
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y P Q R S : C
      f : Quiver.Hom P Q
      g : Quiver.Hom P R
      h : Quiver.Hom Q S
      k : Quiver.Hom R S
      gr : CategoryTheory.RegularEpi g
      comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
      t : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk h  …
      s : CategoryTheory.Limits.Cofork (CategoryTheory.CategoryStruct.comp CategoryT …
      l₁ : Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.RegularEpi.left (Ca …
      l : Quiver.Hom (CategoryTheory.Limits.Cofork.ofπ g ⋯).pt (((CategoryTheory.Fun …
      hl : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ  …
      p : Quiver.Hom (CategoryTheory.Limits.PushoutCocone.mk h k comm).pt (((Categor …
      hp₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCoc …
      right✝ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Pushout …
      m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
      w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ h …
      z : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ h …
      ⊢ ∀ (j : CategoryTheory.Limits.WalkingSpan), Eq (CategoryTheory.CategoryStruct …
    -/
    apply (PushoutCocone.mk _ _ comm).coequalizer_ext
      /-
        case mk.mk.intro.h₀
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        X Y P Q R S : C
        f : Quiver.Hom P Q
        g : Quiver.Hom P R
        h : Quiver.Hom Q S
        k : Quiver.Hom R S
        gr : CategoryTheory.RegularEpi g
        comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
        t : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk h  …
        s : CategoryTheory.Limits.Cofork (CategoryTheory.CategoryStruct.comp CategoryT …
        l₁ : Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.RegularEpi.left (Ca …
        l : Quiver.Hom (CategoryTheory.Limits.Cofork.ofπ g ⋯).pt (((CategoryTheory.Fun …
        hl : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ  …
        p : Quiver.Hom (CategoryTheory.Limits.PushoutCocone.mk h k comm).pt (((Categor …
        hp₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCoc …
        right✝ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Pushout …
        m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
        w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ h …
        z : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ h …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCocone. …
      -/
    · exact z
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.intro.h₁
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        X Y P Q R S : C
        f : Quiver.Hom P Q
        g : Quiver.Hom P R
        h : Quiver.Hom Q S
        k : Quiver.Hom R S
        gr : CategoryTheory.RegularEpi g
        comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
        t : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk h  …
        s : CategoryTheory.Limits.Cofork (CategoryTheory.CategoryStruct.comp CategoryT …
        l₁ : Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.RegularEpi.left (Ca …
        l : Quiver.Hom (CategoryTheory.Limits.Cofork.ofπ g ⋯).pt (((CategoryTheory.Fun …
        hl : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ  …
        p : Quiver.Hom (CategoryTheory.Limits.PushoutCocone.mk h k comm).pt (((Categor …
        hp₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCoc …
        right✝ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Pushout …
        m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
        w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ h …
        z : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ h …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCocone. …
      -/
    · erw [← cancel_epi g, ← Category.assoc, ← eq_whisker comm]
      /-
        case mk.mk.intro.h₁
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        X Y P Q R S : C
        f : Quiver.Hom P Q
        g : Quiver.Hom P R
        h : Quiver.Hom Q S
        k : Quiver.Hom R S
        gr : CategoryTheory.RegularEpi g
        comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
        t : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk h  …
        s : CategoryTheory.Limits.Cofork (CategoryTheory.CategoryStruct.comp CategoryT …
        l₁ : Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.RegularEpi.left (Ca …
        l : Quiver.Hom (CategoryTheory.Limits.Cofork.ofπ g ⋯).pt (((CategoryTheory.Fun …
        hl : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ  …
        p : Quiver.Hom (CategoryTheory.Limits.PushoutCocone.mk h k comm).pt (((Categor …
        hp₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCoc …
        right✝ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Pushout …
        m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
        w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ h …
        z : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ h …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
      -/
      erw [← Category.assoc, ← eq_whisker comm]
      /-
        case mk.mk.intro.h₁
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        X Y P Q R S : C
        f : Quiver.Hom P Q
        g : Quiver.Hom P R
        h : Quiver.Hom Q S
        k : Quiver.Hom R S
        gr : CategoryTheory.RegularEpi g
        comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
        t : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk h  …
        s : CategoryTheory.Limits.Cofork (CategoryTheory.CategoryStruct.comp CategoryT …
        l₁ : Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.RegularEpi.left (Ca …
        l : Quiver.Hom (CategoryTheory.Limits.Cofork.ofπ g ⋯).pt (((CategoryTheory.Fun …
        hl : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ  …
        p : Quiver.Hom (CategoryTheory.Limits.PushoutCocone.mk h k comm).pt (((Categor …
        hp₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCoc …
        right✝ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Pushout …
        m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
        w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ h …
        z : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ h …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
      -/
      dsimp at z; simp only [Category.assoc, z]
                  /-
                    🎉 no goals
                  -/


/-- The first leg of a pushout cocone is a regular epimorphism if the left component is too.

See also `Pushout.fstOfEpi` for the basic epimorphism version, and
`regularOfIsPushoutSndOfRegular` for the flipped version.
-/
def regularOfIsPushoutFstOfRegular {P Q R S : C} {f : P ⟶ Q} {g : P ⟶ R} {h : Q ⟶ S} {k : R ⟶ S}
    [RegularEpi f] (comm : f ≫ h = g ≫ k) (t : IsColimit (PushoutCocone.mk _ _ comm)) :
    RegularEpi k :=
  regularOfIsPushoutSndOfRegular comm.symm (PushoutCocone.flipIsColimit t)


instance (priority := 100) strongEpi_of_regularEpi (f : X ⟶ Y) [RegularEpi f] : StrongEpi f :=
  StrongEpi.mk'
    (by
      /-
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        X Y : C
        f : Quiver.Hom X Y
        inst✝ : CategoryTheory.RegularEpi f
        ⊢ ∀ (X_1 Y_1 : C) (z : Quiver.Hom X_1 Y_1), CategoryTheory.Mono z → ∀ (u : Qui …
      -/
      intro A B z hz u v sq
      have : (RegularEpi.left : RegularEpi.W f ⟶ X) ≫ u = RegularEpi.right ≫ u := by
        apply (cancel_mono z).1
        simp only [Category.assoc, sq.w, RegularEpi.w_assoc]
      /-
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        X Y : C
        f : Quiver.Hom X Y
        inst✝ : CategoryTheory.RegularEpi f
        A B : C
        z : Quiver.Hom A B
        hz : CategoryTheory.Mono z
        u : Quiver.Hom X A
        v : Quiver.Hom Y B
        sq : CategoryTheory.CommSq u f z v
        this : Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.RegularEpi.left u …
        ⊢ sq.HasLift
      -/
      obtain ⟨t, ht⟩ := RegularEpi.desc' f u this
      exact
        CommSq.HasLift.mk'
          ⟨t, ht,
            (cancel_epi f).1
              (by simp only [← Category.assoc, ht, ← sq.w, Arrow.mk_hom, Arrow.homMk'_right])⟩)


/-- A regular epimorphism is an isomorphism if it is a monomorphism. -/
theorem isIso_of_regularEpi_of_mono (f : X ⟶ Y) [RegularEpi f] [Mono f] : IsIso f :=
  isIso_of_mono_of_strongEpi _


/-- A regular epi category is a category in which every epimorphism is regular. -/
class RegularEpiCategory where
  /-- Everyone epimorphism is a regular epimorphism -/
  regularEpiOfEpi : ∀ {X Y : C} (f : X ⟶ Y) [Epi f], RegularEpi f


/-- In a category in which every epimorphism is regular, we can express every epimorphism as
    a coequalizer. This is not an instance because it would create an instance loop. -/
def regularEpiOfEpi [RegularEpiCategory C] (f : X ⟶ Y) [Epi f] : RegularEpi f :=
  RegularEpiCategory.regularEpiOfEpi _


instance (priority := 100) regularEpiCategoryOfSplitEpiCategory [SplitEpiCategory C] :
    RegularEpiCategory C where
  regularEpiOfEpi f _ := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y : C
      inst✝ : CategoryTheory.SplitEpiCategory C
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      x✝ : CategoryTheory.Epi f
      ⊢ CategoryTheory.RegularEpi f
    -/
    haveI := isSplitEpi_of_epi f
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y : C
      inst✝ : CategoryTheory.SplitEpiCategory C
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      x✝ : CategoryTheory.Epi f
      this : CategoryTheory.IsSplitEpi f
      ⊢ CategoryTheory.RegularEpi f
    -/
    infer_instance
    /-
      🎉 no goals
    -/


instance (priority := 100) strongEpiCategory_of_regularEpiCategory [RegularEpiCategory C] :
    StrongEpiCategory C where
  strongEpi_of_epi f _ := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y : C
      inst✝ : CategoryTheory.RegularEpiCategory C
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      x✝ : CategoryTheory.Epi f
      ⊢ CategoryTheory.StrongEpi f
    -/
    haveI := regularEpiOfEpi f
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y : C
      inst✝ : CategoryTheory.RegularEpiCategory C
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      x✝ : CategoryTheory.Epi f
      this : CategoryTheory.RegularEpi f
      ⊢ CategoryTheory.StrongEpi f
    -/
    infer_instance
    /-
      🎉 no goals
    -/


