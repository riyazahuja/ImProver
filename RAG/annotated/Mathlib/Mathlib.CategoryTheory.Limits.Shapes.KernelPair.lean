/-- `IsKernelPair f a b` expresses that `(a, b)` is a kernel pair for `f`, i.e. `a ≫ f = b ≫ f`
and the square
  R → X
  ↓   ↓
  X → Y
is a pullback square.
This is just an abbreviation for `IsPullback a b f f`.
-/
abbrev IsKernelPair :=
  IsPullback a b f f


/-- The data expressing that `(a, b)` is a kernel pair is subsingleton. -/
instance : Subsingleton (IsKernelPair f a b) :=
  ⟨fun P Q => by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      R X Y Z : C
      f : Quiver.Hom X Y
      a b : Quiver.Hom R X
      P Q : CategoryTheory.IsKernelPair f a b
      ⊢ Eq P Q
    -/
    cases P
    /-
      case mk
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      R X Y Z : C
      f : Quiver.Hom X Y
      a b : Quiver.Hom R X
      Q : CategoryTheory.IsKernelPair f a b
      toCommSq✝ : CategoryTheory.CommSq a b f f
      isLimit'✝ : Nonempty (CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Pul …
      ⊢ Eq ⋯ Q
    -/
    cases Q
    /-
      case mk.mk
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      R X Y Z : C
      f : Quiver.Hom X Y
      a b : Quiver.Hom R X
      toCommSq✝¹ : CategoryTheory.CommSq a b f f
      isLimit'✝¹ : Nonempty (CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Pu …
      toCommSq✝ : CategoryTheory.CommSq a b f f
      isLimit'✝ : Nonempty (CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Pul …
      ⊢ Eq ⋯ ⋯
    -/
    congr ⟩
    /-
      🎉 no goals
    -/


/-- If `f` is a monomorphism, then `(𝟙 _, 𝟙 _)` is a kernel pair for `f`. -/
theorem id_of_mono [Mono f] : IsKernelPair f (𝟙 _) (𝟙 _) :=
  ⟨⟨rfl⟩, ⟨PullbackCone.isLimitMkIdId _⟩⟩


instance [Mono f] : Inhabited (IsKernelPair f (𝟙 _) (𝟙 _)) :=
  ⟨id_of_mono f⟩


/--
Given a pair of morphisms `p`, `q` to `X` which factor through `f`, they factor through any kernel
pair of `f`.
-/
noncomputable def lift {S : C} (k : IsKernelPair f a b) (p q : S ⟶ X) (w : p ≫ f = q ≫ f) :
    S ⟶ R :=
  PullbackCone.IsLimit.lift k.isLimit _ _ w


@[reassoc (attr := simp)]
lemma lift_fst {S : C} (k : IsKernelPair f a b) (p q : S ⟶ X) (w : p ≫ f = q ≫ f) :
    k.lift p q w ≫ a = p :=
  PullbackCone.IsLimit.lift_fst _ _ _ _


@[reassoc (attr := simp)]
lemma lift_snd {S : C} (k : IsKernelPair f a b) (p q : S ⟶ X) (w : p ≫ f = q ≫ f) :
    k.lift p q w ≫ b = q :=
  PullbackCone.IsLimit.lift_snd _ _ _ _


/--
Given a pair of morphisms `p`, `q` to `X` which factor through `f`, they factor through any kernel
pair of `f`.
-/
noncomputable def lift' {S : C} (k : IsKernelPair f a b) (p q : S ⟶ X) (w : p ≫ f = q ≫ f) :
    { t : S ⟶ R // t ≫ a = p ∧ t ≫ b = q } :=
                    /-
                      C : Type u
                      inst✝ : CategoryTheory.Category.{v, u} C
                      R X Y Z : C
                      f : Quiver.Hom X Y
                      a b : Quiver.Hom R X
                      S : C
                      k : CategoryTheory.IsKernelPair f a b
                      p q : Quiver.Hom S X
                      w : Eq (CategoryTheory.CategoryStruct.comp p f) (CategoryTheory.CategoryStruct …
                      ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (k.lift p q w) a) p) (Eq (Catego …
                    -/
  ⟨k.lift p q w, by simp⟩
                    /-
                      🎉 no goals
                    -/


/--
If `(a,b)` is a kernel pair for `f₁ ≫ f₂` and `a ≫ f₁ = b ≫ f₁`, then `(a,b)` is a kernel pair for
just `f₁`.
That is, to show that `(a,b)` is a kernel pair for `f₁` it suffices to only show the square
commutes, rather than to additionally show it's a pullback.
-/
theorem cancel_right {f₁ : X ⟶ Y} {f₂ : Y ⟶ Z} (comm : a ≫ f₁ = b ≫ f₁)
    (big_k : IsKernelPair (f₁ ≫ f₂) a b) : IsKernelPair f₁ a b :=
  { w := comm
    isLimit' :=
      ⟨PullbackCone.isLimitAux' _ fun s => by
        let s' : PullbackCone (f₁ ≫ f₂) (f₁ ≫ f₂) :=
          PullbackCone.mk s.fst s.snd (s.condition_assoc _)
        refine ⟨big_k.isLimit.lift s', big_k.isLimit.fac _ WalkingCospan.left,
          big_k.isLimit.fac _ WalkingCospan.right, fun m₁ m₂ => ?_⟩
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          R X Y Z : C
          a b : Quiver.Hom R X
          f₁ : Quiver.Hom X Y
          f₂ : Quiver.Hom Y Z
          comm : Eq (CategoryTheory.CategoryStruct.comp a f₁) (CategoryTheory.CategorySt …
          big_k : CategoryTheory.IsKernelPair (CategoryTheory.CategoryStruct.comp f₁ f₂) …
          s : CategoryTheory.Limits.PullbackCone f₁ f₁
          s' : CategoryTheory.Limits.PullbackCone (CategoryTheory.CategoryStruct.comp f₁ …
          m✝ : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk a b ⋯).pt
          m₁ : Eq (CategoryTheory.CategoryStruct.comp m✝ (CategoryTheory.Limits.Pullback …
          m₂ : Eq (CategoryTheory.CategoryStruct.comp m✝ (CategoryTheory.Limits.Pullback …
          ⊢ Eq m✝ ((CategoryTheory.IsPullback.isLimit big_k).lift s')
        -/
        apply big_k.isLimit.hom_ext
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          R X Y Z : C
          a b : Quiver.Hom R X
          f₁ : Quiver.Hom X Y
          f₂ : Quiver.Hom Y Z
          comm : Eq (CategoryTheory.CategoryStruct.comp a f₁) (CategoryTheory.CategorySt …
          big_k : CategoryTheory.IsKernelPair (CategoryTheory.CategoryStruct.comp f₁ f₂) …
          s : CategoryTheory.Limits.PullbackCone f₁ f₁
          s' : CategoryTheory.Limits.PullbackCone (CategoryTheory.CategoryStruct.comp f₁ …
          m✝ : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk a b ⋯).pt
          m₁ : Eq (CategoryTheory.CategoryStruct.comp m✝ (CategoryTheory.Limits.Pullback …
          m₂ : Eq (CategoryTheory.CategoryStruct.comp m✝ (CategoryTheory.Limits.Pullback …
          ⊢ ∀ (j : CategoryTheory.Limits.WalkingCospan), Eq (CategoryTheory.CategoryStru …
        -/
        refine (PullbackCone.mk a b ?_ : PullbackCone (f₁ ≫ f₂) _).equalizer_ext ?_ ?_
          /-
            case refine_1
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            R X Y Z : C
            a b : Quiver.Hom R X
            f₁ : Quiver.Hom X Y
            f₂ : Quiver.Hom Y Z
            comm : Eq (CategoryTheory.CategoryStruct.comp a f₁) (CategoryTheory.CategorySt …
            big_k : CategoryTheory.IsKernelPair (CategoryTheory.CategoryStruct.comp f₁ f₂) …
            s : CategoryTheory.Limits.PullbackCone f₁ f₁
            s' : CategoryTheory.Limits.PullbackCone (CategoryTheory.CategoryStruct.comp f₁ …
            m✝ : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk a b ⋯).pt
            m₁ : Eq (CategoryTheory.CategoryStruct.comp m✝ (CategoryTheory.Limits.Pullback …
            m₂ : Eq (CategoryTheory.CategoryStruct.comp m✝ (CategoryTheory.Limits.Pullback …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp a (CategoryTheory.CategoryStruct.comp …
          -/
        · apply reassoc_of% comm
          /-
            🎉 no goals
          -/
          /-
            case refine_2
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            R X Y Z : C
            a b : Quiver.Hom R X
            f₁ : Quiver.Hom X Y
            f₂ : Quiver.Hom Y Z
            comm : Eq (CategoryTheory.CategoryStruct.comp a f₁) (CategoryTheory.CategorySt …
            big_k : CategoryTheory.IsKernelPair (CategoryTheory.CategoryStruct.comp f₁ f₂) …
            s : CategoryTheory.Limits.PullbackCone f₁ f₁
            s' : CategoryTheory.Limits.PullbackCone (CategoryTheory.CategoryStruct.comp f₁ …
            m✝ : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk a b ⋯).pt
            m₁ : Eq (CategoryTheory.CategoryStruct.comp m✝ (CategoryTheory.Limits.Pullback …
            m₂ : Eq (CategoryTheory.CategoryStruct.comp m✝ (CategoryTheory.Limits.Pullback …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp m✝ (CategoryTheory.Limits.PullbackCon …
          -/
        · apply m₁.trans (big_k.isLimit.fac s' WalkingCospan.left).symm
          /-
            🎉 no goals
          -/
          /-
            case refine_3
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            R X Y Z : C
            a b : Quiver.Hom R X
            f₁ : Quiver.Hom X Y
            f₂ : Quiver.Hom Y Z
            comm : Eq (CategoryTheory.CategoryStruct.comp a f₁) (CategoryTheory.CategorySt …
            big_k : CategoryTheory.IsKernelPair (CategoryTheory.CategoryStruct.comp f₁ f₂) …
            s : CategoryTheory.Limits.PullbackCone f₁ f₁
            s' : CategoryTheory.Limits.PullbackCone (CategoryTheory.CategoryStruct.comp f₁ …
            m✝ : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk a b ⋯).pt
            m₁ : Eq (CategoryTheory.CategoryStruct.comp m✝ (CategoryTheory.Limits.Pullback …
            m₂ : Eq (CategoryTheory.CategoryStruct.comp m✝ (CategoryTheory.Limits.Pullback …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp m✝ (CategoryTheory.Limits.PullbackCon …
          -/
        · apply m₂.trans (big_k.isLimit.fac s' WalkingCospan.right).symm⟩ }
          /-
            🎉 no goals
          -/


/-- If `(a,b)` is a kernel pair for `f₁ ≫ f₂` and `f₂` is mono, then `(a,b)` is a kernel pair for
just `f₁`.
The converse of `comp_of_mono`.
-/
theorem cancel_right_of_mono {f₁ : X ⟶ Y} {f₂ : Y ⟶ Z} [Mono f₂]
    (big_k : IsKernelPair (f₁ ≫ f₂) a b) : IsKernelPair f₁ a b :=
                   /-
                     C : Type u
                     inst✝¹ : CategoryTheory.Category.{v, u} C
                     R X Y Z : C
                     a b : Quiver.Hom R X
                     f₁ : Quiver.Hom X Y
                     f₂ : Quiver.Hom Y Z
                     inst✝ : CategoryTheory.Mono f₂
                     big_k : CategoryTheory.IsKernelPair (CategoryTheory.CategoryStruct.comp f₁ f₂) …
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp a f₁) (CategoryTheory.CategoryStruct. …
                   -/
  cancel_right (by rw [← cancel_mono f₂, assoc, assoc, big_k.w]) big_k
                   /-
                     🎉 no goals
                   -/


/--
If `(a,b)` is a kernel pair for `f₁` and `f₂` is mono, then `(a,b)` is a kernel pair for `f₁ ≫ f₂`.
The converse of `cancel_right_of_mono`.
-/
theorem comp_of_mono {f₁ : X ⟶ Y} {f₂ : Y ⟶ Z} [Mono f₂] (small_k : IsKernelPair f₁ a b) :
    IsKernelPair (f₁ ≫ f₂) a b :=
            /-
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              R X Y Z : C
              a b : Quiver.Hom R X
              f₁ : Quiver.Hom X Y
              f₂ : Quiver.Hom Y Z
              inst✝ : CategoryTheory.Mono f₂
              small_k : CategoryTheory.IsKernelPair f₁ a b
              ⊢ Eq (CategoryTheory.CategoryStruct.comp a (CategoryTheory.CategoryStruct.comp …
            -/
  { w := by rw [small_k.w_assoc]
            /-
              🎉 no goals
            -/
    isLimit' := ⟨by
      refine PullbackCone.isLimitAux _
        (fun s => small_k.lift s.fst s.snd (by rw [← cancel_mono f₂, assoc, s.condition, assoc]))
        (by simp) (by simp) ?_
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        R X Y Z : C
        a b : Quiver.Hom R X
        f₁ : Quiver.Hom X Y
        f₂ : Quiver.Hom Y Z
        inst✝ : CategoryTheory.Mono f₂
        small_k : CategoryTheory.IsKernelPair f₁ a b
        ⊢ ∀ (s : CategoryTheory.Limits.PullbackCone (CategoryTheory.CategoryStruct.com …
      -/
      intro s m hm
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        R X Y Z : C
        a b : Quiver.Hom R X
        f₁ : Quiver.Hom X Y
        f₂ : Quiver.Hom Y Z
        inst✝ : CategoryTheory.Mono f₂
        small_k : CategoryTheory.IsKernelPair f₁ a b
        s : CategoryTheory.Limits.PullbackCone (CategoryTheory.CategoryStruct.comp f₁  …
        m : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk a b ⋯).pt
        hm : ∀ (j : CategoryTheory.Limits.WalkingCospan), Eq (CategoryTheory.CategoryS …
        ⊢ Eq m ((fun s => small_k.lift s.fst s.snd ⋯) s)
      -/
      apply small_k.isLimit.hom_ext
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        R X Y Z : C
        a b : Quiver.Hom R X
        f₁ : Quiver.Hom X Y
        f₂ : Quiver.Hom Y Z
        inst✝ : CategoryTheory.Mono f₂
        small_k : CategoryTheory.IsKernelPair f₁ a b
        s : CategoryTheory.Limits.PullbackCone (CategoryTheory.CategoryStruct.comp f₁  …
        m : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk a b ⋯).pt
        hm : ∀ (j : CategoryTheory.Limits.WalkingCospan), Eq (CategoryTheory.CategoryS …
        ⊢ ∀ (j : CategoryTheory.Limits.WalkingCospan), Eq (CategoryTheory.CategoryStru …
      -/
      apply PullbackCone.equalizer_ext small_k.cone _ _
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          R X Y Z : C
          a b : Quiver.Hom R X
          f₁ : Quiver.Hom X Y
          f₂ : Quiver.Hom Y Z
          inst✝ : CategoryTheory.Mono f₂
          small_k : CategoryTheory.IsKernelPair f₁ a b
          s : CategoryTheory.Limits.PullbackCone (CategoryTheory.CategoryStruct.comp f₁  …
          m : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk a b ⋯).pt
          hm : ∀ (j : CategoryTheory.Limits.WalkingCospan), Eq (CategoryTheory.CategoryS …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.IsPullback.cone sma …
        -/
      · exact (hm WalkingCospan.left).trans (by simp)
        /-
          🎉 no goals
        -/
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          R X Y Z : C
          a b : Quiver.Hom R X
          f₁ : Quiver.Hom X Y
          f₂ : Quiver.Hom Y Z
          inst✝ : CategoryTheory.Mono f₂
          small_k : CategoryTheory.IsKernelPair f₁ a b
          s : CategoryTheory.Limits.PullbackCone (CategoryTheory.CategoryStruct.comp f₁  …
          m : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk a b ⋯).pt
          hm : ∀ (j : CategoryTheory.Limits.WalkingCospan), Eq (CategoryTheory.CategoryS …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.IsPullback.cone sma …
        -/
      · exact (hm WalkingCospan.right).trans (by simp)⟩ }
        /-
          🎉 no goals
        -/


/--
If `(a,b)` is the kernel pair of `f`, and `f` is a coequalizer morphism for some parallel pair, then
`f` is a coequalizer morphism of `a` and `b`.
-/
def toCoequalizer (k : IsKernelPair f a b) [r : RegularEpi f] : IsColimit (Cofork.ofπ f k.w) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    R X Y Z : C
    f : Quiver.Hom X Y
    a b : Quiver.Hom R X
    k : CategoryTheory.IsKernelPair f a b
    r : CategoryTheory.RegularEpi f
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofork.ofπ f ⋯)
  -/
  let t := k.isLimit.lift (PullbackCone.mk _ _ r.w)
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    R X Y Z : C
    f : Quiver.Hom X Y
    a b : Quiver.Hom R X
    k : CategoryTheory.IsKernelPair f a b
    r : CategoryTheory.RegularEpi f
    t : Quiver.Hom (CategoryTheory.Limits.PullbackCone.mk CategoryTheory.RegularEp …
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofork.ofπ f ⋯)
  -/
  have ht : t ≫ a = r.left := k.isLimit.fac _ WalkingCospan.left
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    R X Y Z : C
    f : Quiver.Hom X Y
    a b : Quiver.Hom R X
    k : CategoryTheory.IsKernelPair f a b
    r : CategoryTheory.RegularEpi f
    t : Quiver.Hom (CategoryTheory.Limits.PullbackCone.mk CategoryTheory.RegularEp …
    ht : Eq (CategoryTheory.CategoryStruct.comp t a) CategoryTheory.RegularEpi.left
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofork.ofπ f ⋯)
  -/
  have kt : t ≫ b = r.right := k.isLimit.fac _ WalkingCospan.right
  refine Cofork.IsColimit.mk _
    (fun s => Cofork.IsColimit.desc r.isColimit s.π
      (by rw [← ht, assoc, s.condition, reassoc_of% kt]))
    (fun s => ?_) (fun s m w => ?_)
    /-
      case refine_1
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      R X Y Z : C
      f : Quiver.Hom X Y
      a b : Quiver.Hom R X
      k : CategoryTheory.IsKernelPair f a b
      r : CategoryTheory.RegularEpi f
      t : Quiver.Hom (CategoryTheory.Limits.PullbackCone.mk CategoryTheory.RegularEp …
      ht : Eq (CategoryTheory.CategoryStruct.comp t a) CategoryTheory.RegularEpi.left
      kt : Eq (CategoryTheory.CategoryStruct.comp t b) CategoryTheory.RegularEpi.right
      s : CategoryTheory.Limits.Cofork a b
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ f ⋯ …
    -/
  · apply Cofork.IsColimit.π_desc' r.isColimit
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      R X Y Z : C
      f : Quiver.Hom X Y
      a b : Quiver.Hom R X
      k : CategoryTheory.IsKernelPair f a b
      r : CategoryTheory.RegularEpi f
      t : Quiver.Hom (CategoryTheory.Limits.PullbackCone.mk CategoryTheory.RegularEp …
      ht : Eq (CategoryTheory.CategoryStruct.comp t a) CategoryTheory.RegularEpi.left
      kt : Eq (CategoryTheory.CategoryStruct.comp t b) CategoryTheory.RegularEpi.right
      s : CategoryTheory.Limits.Cofork a b
      m : Quiver.Hom (CategoryTheory.Limits.Cofork.ofπ f ⋯).pt s.pt
      w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ f …
      ⊢ Eq m ((fun s => CategoryTheory.Limits.Cofork.IsColimit.desc CategoryTheory.R …
    -/
  · apply Cofork.IsColimit.hom_ext r.isColimit
    /-
      case refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      R X Y Z : C
      f : Quiver.Hom X Y
      a b : Quiver.Hom R X
      k : CategoryTheory.IsKernelPair f a b
      r : CategoryTheory.RegularEpi f
      t : Quiver.Hom (CategoryTheory.Limits.PullbackCone.mk CategoryTheory.RegularEp …
      ht : Eq (CategoryTheory.CategoryStruct.comp t a) CategoryTheory.RegularEpi.left
      kt : Eq (CategoryTheory.CategoryStruct.comp t b) CategoryTheory.RegularEpi.right
      s : CategoryTheory.Limits.Cofork a b
      m : Quiver.Hom (CategoryTheory.Limits.Cofork.ofπ f ⋯).pt s.pt
      w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ f …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ f ⋯ …
    -/
    exact w.trans (Cofork.IsColimit.π_desc' r.isColimit _ _).symm
    /-
      🎉 no goals
    -/


/-- If `a₁ a₂ : A ⟶ Y` is a kernel pair for `g : Y ⟶ Z`, then `a₁ ×[Z] X` and `a₂ ×[Z] X`
(`A ×[Z] X ⟶ Y ×[Z] X`) is a kernel pair for `Y ×[Z] X ⟶ X`. -/
protected theorem pullback {X Y Z A : C} {g : Y ⟶ Z} {a₁ a₂ : A ⟶ Y} (h : IsKernelPair g a₁ a₂)
    (f : X ⟶ Z) [HasPullback f g] [HasPullback f (a₁ ≫ g)] :
    IsKernelPair (pullback.fst f g)
                                               /-
                                                 C : Type u
                                                 inst✝² : CategoryTheory.Category.{v, u} C
                                                 R X✝ Y✝ Z✝ : C
                                                 f✝ : Quiver.Hom X✝ Y✝
                                                 a b : Quiver.Hom R X✝
                                                 X Y Z A : C
                                                 g : Quiver.Hom Y Z
                                                 a₁ a₂ : Quiver.Hom A Y
                                                 h : CategoryTheory.IsKernelPair g a₁ a₂
                                                 f : Quiver.Hom X Z
                                                 inst✝¹ : CategoryTheory.Limits.HasPullback f g
                                                 inst✝ : CategoryTheory.Limits.HasPullback f (CategoryTheory.CategoryStruct.com …
                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id Z …
                                               -/
      (pullback.map f _ f _ (𝟙 X) a₁ (𝟙 Z) (by simp) <| Category.comp_id _)
                                               /-
                                                 🎉 no goals
                                               -/
                                               /-
                                                 C : Type u
                                                 inst✝² : CategoryTheory.Category.{v, u} C
                                                 R X✝ Y✝ Z✝ : C
                                                 f✝ : Quiver.Hom X✝ Y✝
                                                 a b : Quiver.Hom R X✝
                                                 X Y Z A : C
                                                 g : Quiver.Hom Y Z
                                                 a₁ a₂ : Quiver.Hom A Y
                                                 h : CategoryTheory.IsKernelPair g a₁ a₂
                                                 f : Quiver.Hom X Z
                                                 inst✝¹ : CategoryTheory.Limits.HasPullback f g
                                                 inst✝ : CategoryTheory.Limits.HasPullback f (CategoryTheory.CategoryStruct.com …
                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id Z …
                                               -/
      (pullback.map _ _ _ _ (𝟙 X) a₂ (𝟙 Z) (by simp) <| (Category.comp_id _).trans h.1.1) := by
                                               /-
                                                 🎉 no goals
                                               -/
  refine ⟨⟨by rw [pullback.lift_fst, pullback.lift_fst]⟩, ⟨PullbackCone.isLimitAux _
    (fun s => pullback.lift (s.fst ≫ pullback.fst _ _)
      (h.lift (s.fst ≫ pullback.snd _ _) (s.snd ≫ pullback.snd _ _) ?_ ) ?_) (fun s => ?_)
        (fun s => ?_) (fun s m hm => ?_)⟩⟩
    /-
      case refine_1
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      X Y Z A : C
      g : Quiver.Hom Y Z
      a₁ a₂ : Quiver.Hom A Y
      h : CategoryTheory.IsKernelPair g a₁ a₂
      f : Quiver.Hom X Z
      inst✝¹ : CategoryTheory.Limits.HasPullback f g
      inst✝ : CategoryTheory.Limits.HasPullback f (CategoryTheory.CategoryStruct.com …
      s : CategoryTheory.Limits.PullbackCone (CategoryTheory.Limits.pullback.fst f g …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp s …
    -/
  · simp_rw [Category.assoc, ← pullback.condition, ← Category.assoc, s.condition]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      X Y Z A : C
      g : Quiver.Hom Y Z
      a₁ a₂ : Quiver.Hom A Y
      h : CategoryTheory.IsKernelPair g a₁ a₂
      f : Quiver.Hom X Z
      inst✝¹ : CategoryTheory.Limits.HasPullback f g
      inst✝ : CategoryTheory.Limits.HasPullback f (CategoryTheory.CategoryStruct.com …
      s : CategoryTheory.Limits.PullbackCone (CategoryTheory.Limits.pullback.fst f g …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp s …
    -/
  · simp only [assoc, lift_fst_assoc, pullback.condition]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      X Y Z A : C
      g : Quiver.Hom Y Z
      a₁ a₂ : Quiver.Hom A Y
      h : CategoryTheory.IsKernelPair g a₁ a₂
      f : Quiver.Hom X Z
      inst✝¹ : CategoryTheory.Limits.HasPullback f g
      inst✝ : CategoryTheory.Limits.HasPullback f (CategoryTheory.CategoryStruct.com …
      s : CategoryTheory.Limits.PullbackCone (CategoryTheory.Limits.pullback.fst f g …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => CategoryTheory.Limits.pull …
    -/
            /-
              🎉 no goals
            -/
  · ext <;> simp
            /-
              🎉 no goals
            -/
    /-
      case refine_4
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      X Y Z A : C
      g : Quiver.Hom Y Z
      a₁ a₂ : Quiver.Hom A Y
      h : CategoryTheory.IsKernelPair g a₁ a₂
      f : Quiver.Hom X Z
      inst✝¹ : CategoryTheory.Limits.HasPullback f g
      inst✝ : CategoryTheory.Limits.HasPullback f (CategoryTheory.CategoryStruct.com …
      s : CategoryTheory.Limits.PullbackCone (CategoryTheory.Limits.pullback.fst f g …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => CategoryTheory.Limits.pull …
    -/
  · ext
      /-
        case refine_4.h₀
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        X Y Z A : C
        g : Quiver.Hom Y Z
        a₁ a₂ : Quiver.Hom A Y
        h : CategoryTheory.IsKernelPair g a₁ a₂
        f : Quiver.Hom X Z
        inst✝¹ : CategoryTheory.Limits.HasPullback f g
        inst✝ : CategoryTheory.Limits.HasPullback f (CategoryTheory.CategoryStruct.com …
        s : CategoryTheory.Limits.PullbackCone (CategoryTheory.Limits.pullback.fst f g …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simp [s.condition]
      /-
        🎉 no goals
      -/
      /-
        case refine_4.h₁
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        X Y Z A : C
        g : Quiver.Hom Y Z
        a₁ a₂ : Quiver.Hom A Y
        h : CategoryTheory.IsKernelPair g a₁ a₂
        f : Quiver.Hom X Z
        inst✝¹ : CategoryTheory.Limits.HasPullback f g
        inst✝ : CategoryTheory.Limits.HasPullback f (CategoryTheory.CategoryStruct.com …
        s : CategoryTheory.Limits.PullbackCone (CategoryTheory.Limits.pullback.fst f g …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simp
      /-
        🎉 no goals
      -/
  · #adaptation_note /-- nightly-2024-04-01
    This `symm` (or the following ones that undo it) wasn't previously necessary. -/
    /-
      case refine_5
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      X Y Z A : C
      g : Quiver.Hom Y Z
      a₁ a₂ : Quiver.Hom A Y
      h : CategoryTheory.IsKernelPair g a₁ a₂
      f : Quiver.Hom X Z
      inst✝¹ : CategoryTheory.Limits.HasPullback f g
      inst✝ : CategoryTheory.Limits.HasPullback f (CategoryTheory.CategoryStruct.com …
      s : CategoryTheory.Limits.PullbackCone (CategoryTheory.Limits.pullback.fst f g …
      m : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk (CategoryTheory.Lim …
      hm : ∀ (j : CategoryTheory.Limits.WalkingCospan), Eq (CategoryTheory.CategoryS …
      ⊢ Eq m ((fun s => CategoryTheory.Limits.pullback.lift (CategoryTheory.Category …
    -/
    symm
    /-
      case refine_5
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      X Y Z A : C
      g : Quiver.Hom Y Z
      a₁ a₂ : Quiver.Hom A Y
      h : CategoryTheory.IsKernelPair g a₁ a₂
      f : Quiver.Hom X Z
      inst✝¹ : CategoryTheory.Limits.HasPullback f g
      inst✝ : CategoryTheory.Limits.HasPullback f (CategoryTheory.CategoryStruct.com …
      s : CategoryTheory.Limits.PullbackCone (CategoryTheory.Limits.pullback.fst f g …
      m : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk (CategoryTheory.Lim …
      hm : ∀ (j : CategoryTheory.Limits.WalkingCospan), Eq (CategoryTheory.CategoryS …
      ⊢ Eq ((fun s => CategoryTheory.Limits.pullback.lift (CategoryTheory.CategorySt …
    -/
    apply pullback.hom_ext
      /-
        case refine_5.h₀
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        X Y Z A : C
        g : Quiver.Hom Y Z
        a₁ a₂ : Quiver.Hom A Y
        h : CategoryTheory.IsKernelPair g a₁ a₂
        f : Quiver.Hom X Z
        inst✝¹ : CategoryTheory.Limits.HasPullback f g
        inst✝ : CategoryTheory.Limits.HasPullback f (CategoryTheory.CategoryStruct.com …
        s : CategoryTheory.Limits.PullbackCone (CategoryTheory.Limits.pullback.fst f g …
        m : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk (CategoryTheory.Lim …
        hm : ∀ (j : CategoryTheory.Limits.WalkingCospan), Eq (CategoryTheory.CategoryS …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => CategoryTheory.Limits.pull …
      -/
    · symm
      /-
        case refine_5.h₀
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        X Y Z A : C
        g : Quiver.Hom Y Z
        a₁ a₂ : Quiver.Hom A Y
        h : CategoryTheory.IsKernelPair g a₁ a₂
        f : Quiver.Hom X Z
        inst✝¹ : CategoryTheory.Limits.HasPullback f g
        inst✝ : CategoryTheory.Limits.HasPullback f (CategoryTheory.CategoryStruct.com …
        s : CategoryTheory.Limits.PullbackCone (CategoryTheory.Limits.pullback.fst f g …
        m : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk (CategoryTheory.Lim …
        hm : ∀ (j : CategoryTheory.Limits.WalkingCospan), Eq (CategoryTheory.CategoryS …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.pullback.fst …
      -/
      simpa using hm WalkingCospan.left =≫ pullback.fst f g
      /-
        🎉 no goals
      -/
      /-
        case refine_5.h₁
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        X Y Z A : C
        g : Quiver.Hom Y Z
        a₁ a₂ : Quiver.Hom A Y
        h : CategoryTheory.IsKernelPair g a₁ a₂
        f : Quiver.Hom X Z
        inst✝¹ : CategoryTheory.Limits.HasPullback f g
        inst✝ : CategoryTheory.Limits.HasPullback f (CategoryTheory.CategoryStruct.com …
        s : CategoryTheory.Limits.PullbackCone (CategoryTheory.Limits.pullback.fst f g …
        m : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk (CategoryTheory.Lim …
        hm : ∀ (j : CategoryTheory.Limits.WalkingCospan), Eq (CategoryTheory.CategoryS …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => CategoryTheory.Limits.pull …
      -/
    · symm
      /-
        case refine_5.h₁
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        X Y Z A : C
        g : Quiver.Hom Y Z
        a₁ a₂ : Quiver.Hom A Y
        h : CategoryTheory.IsKernelPair g a₁ a₂
        f : Quiver.Hom X Z
        inst✝¹ : CategoryTheory.Limits.HasPullback f g
        inst✝ : CategoryTheory.Limits.HasPullback f (CategoryTheory.CategoryStruct.com …
        s : CategoryTheory.Limits.PullbackCone (CategoryTheory.Limits.pullback.fst f g …
        m : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk (CategoryTheory.Lim …
        hm : ∀ (j : CategoryTheory.Limits.WalkingCospan), Eq (CategoryTheory.CategoryS …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.pullback.snd …
      -/
      apply PullbackCone.IsLimit.hom_ext h.isLimit
        /-
          case refine_5.h₁.h₀
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          X Y Z A : C
          g : Quiver.Hom Y Z
          a₁ a₂ : Quiver.Hom A Y
          h : CategoryTheory.IsKernelPair g a₁ a₂
          f : Quiver.Hom X Z
          inst✝¹ : CategoryTheory.Limits.HasPullback f g
          inst✝ : CategoryTheory.Limits.HasPullback f (CategoryTheory.CategoryStruct.com …
          s : CategoryTheory.Limits.PullbackCone (CategoryTheory.Limits.pullback.fst f g …
          m : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk (CategoryTheory.Lim …
          hm : ∀ (j : CategoryTheory.Limits.WalkingCospan), Eq (CategoryTheory.CategoryS …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp m …
        -/
      · simpa using hm WalkingCospan.left =≫ pullback.snd f g
        /-
          🎉 no goals
        -/
        /-
          case refine_5.h₁.h₁
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          X Y Z A : C
          g : Quiver.Hom Y Z
          a₁ a₂ : Quiver.Hom A Y
          h : CategoryTheory.IsKernelPair g a₁ a₂
          f : Quiver.Hom X Z
          inst✝¹ : CategoryTheory.Limits.HasPullback f g
          inst✝ : CategoryTheory.Limits.HasPullback f (CategoryTheory.CategoryStruct.com …
          s : CategoryTheory.Limits.PullbackCone (CategoryTheory.Limits.pullback.fst f g …
          m : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk (CategoryTheory.Lim …
          hm : ∀ (j : CategoryTheory.Limits.WalkingCospan), Eq (CategoryTheory.CategoryS …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp m …
        -/
      · simpa using hm WalkingCospan.right =≫ pullback.snd f g
        /-
          🎉 no goals
        -/


theorem mono_of_isIso_fst (h : IsKernelPair f a b) [IsIso a] : Mono f := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    R X Y : C
    f : Quiver.Hom X Y
    a b : Quiver.Hom R X
    h : CategoryTheory.IsKernelPair f a b
    inst✝ : CategoryTheory.IsIso a
    ⊢ CategoryTheory.Mono f
  -/
  obtain ⟨l, h₁, h₂⟩ := Limits.PullbackCone.IsLimit.lift' h.isLimit (𝟙 _) (𝟙 _) (by simp [h.w])
  /-
    case mk.intro
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    R X Y : C
    f : Quiver.Hom X Y
    a b : Quiver.Hom R X
    h : CategoryTheory.IsKernelPair f a b
    inst✝ : CategoryTheory.IsIso a
    l : Quiver.Hom X (CategoryTheory.IsPullback.cone h).pt
    h₁ : Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheory.IsPullback.cone  …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheory.IsPullback.cone  …
    ⊢ CategoryTheory.Mono f
  -/
  rw [IsPullback.cone_fst, ← IsIso.eq_comp_inv, Category.id_comp] at h₁
  /-
    case mk.intro
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    R X Y : C
    f : Quiver.Hom X Y
    a b : Quiver.Hom R X
    h : CategoryTheory.IsKernelPair f a b
    inst✝ : CategoryTheory.IsIso a
    l : Quiver.Hom X (CategoryTheory.IsPullback.cone h).pt
    h₁ : Eq l (CategoryTheory.inv a)
    h₂ : Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheory.IsPullback.cone  …
    ⊢ CategoryTheory.Mono f
  -/
  rw [h₁, IsIso.inv_comp_eq, Category.comp_id] at h₂
  /-
    case mk.intro
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    R X Y : C
    f : Quiver.Hom X Y
    a b : Quiver.Hom R X
    h : CategoryTheory.IsKernelPair f a b
    inst✝ : CategoryTheory.IsIso a
    l : Quiver.Hom X (CategoryTheory.IsPullback.cone h).pt
    h₁ : Eq l (CategoryTheory.inv a)
    h₂ : Eq (CategoryTheory.IsPullback.cone h).snd a
    ⊢ CategoryTheory.Mono f
  -/
  constructor
  /-
    case mk.intro.right_cancellation
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    R X Y : C
    f : Quiver.Hom X Y
    a b : Quiver.Hom R X
    h : CategoryTheory.IsKernelPair f a b
    inst✝ : CategoryTheory.IsIso a
    l : Quiver.Hom X (CategoryTheory.IsPullback.cone h).pt
    h₁ : Eq l (CategoryTheory.inv a)
    h₂ : Eq (CategoryTheory.IsPullback.cone h).snd a
    ⊢ ∀ {Z : C} (g h : Quiver.Hom Z X), Eq (CategoryTheory.CategoryStruct.comp g f …
  -/
  intro Z g₁ g₂ e
  /-
    case mk.intro.right_cancellation
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    R X Y : C
    f : Quiver.Hom X Y
    a b : Quiver.Hom R X
    h : CategoryTheory.IsKernelPair f a b
    inst✝ : CategoryTheory.IsIso a
    l : Quiver.Hom X (CategoryTheory.IsPullback.cone h).pt
    h₁ : Eq l (CategoryTheory.inv a)
    h₂ : Eq (CategoryTheory.IsPullback.cone h).snd a
    Z : C
    g₁ g₂ : Quiver.Hom Z X
    e : Eq (CategoryTheory.CategoryStruct.comp g₁ f) (CategoryTheory.CategoryStruc …
    ⊢ Eq g₁ g₂
  -/
  obtain ⟨l', rfl, rfl⟩ := Limits.PullbackCone.IsLimit.lift' h.isLimit _ _ e
  /-
    case mk.intro.right_cancellation.mk.intro
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    R X Y : C
    f : Quiver.Hom X Y
    a b : Quiver.Hom R X
    h : CategoryTheory.IsKernelPair f a b
    inst✝ : CategoryTheory.IsIso a
    l : Quiver.Hom X (CategoryTheory.IsPullback.cone h).pt
    h₁ : Eq l (CategoryTheory.inv a)
    h₂ : Eq (CategoryTheory.IsPullback.cone h).snd a
    Z : C
    l' : Quiver.Hom Z (CategoryTheory.IsPullback.cone h).pt
    e : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp l' (CategoryTheory.IsPullback.cone h) …
  -/
  rw [IsPullback.cone_fst, h₂]
  /-
    🎉 no goals
  -/


theorem isIso_of_mono (h : IsKernelPair f a b) [Mono f] : IsIso a := by
  rw [←
    show _ = a from
      (Category.comp_id _).symm.trans
        ((IsKernelPair.id_of_mono f).isLimit.conePointUniqueUpToIso_inv_comp h.isLimit
          WalkingCospan.left)]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    R X Y : C
    f : Quiver.Hom X Y
    a b : Quiver.Hom R X
    h : CategoryTheory.IsKernelPair f a b
    inst✝ : CategoryTheory.Mono f
    ⊢ CategoryTheory.IsIso ((CategoryTheory.IsPullback.isLimit ⋯).conePointUniqueU …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem of_isIso_of_mono [IsIso a] [Mono f] : IsKernelPair f a a := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    R X Y : C
    f : Quiver.Hom X Y
    a : Quiver.Hom R X
    inst✝¹ : CategoryTheory.IsIso a
    inst✝ : CategoryTheory.Mono f
    ⊢ CategoryTheory.IsKernelPair f a a
  -/
  change IsPullback _ _ _ _
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    R X Y : C
    f : Quiver.Hom X Y
    a : Quiver.Hom R X
    inst✝¹ : CategoryTheory.IsIso a
    inst✝ : CategoryTheory.Mono f
    ⊢ CategoryTheory.IsPullback a a f f
  -/
  convert (IsPullback.of_horiz_isIso ⟨(rfl : a ≫ 𝟙 X = _ )⟩).paste_vert (IsKernelPair.id_of_mono f)
  /-
    case h.e'_8
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    R X Y : C
    f : Quiver.Hom X Y
    a : Quiver.Hom R X
    inst✝¹ : CategoryTheory.IsIso a
    inst✝ : CategoryTheory.Mono f
    ⊢ Eq a (CategoryTheory.CategoryStruct.comp a (CategoryTheory.CategoryStruct.id …
  -/
  all_goals { simp }
  /-
    🎉 no goals
  -/


