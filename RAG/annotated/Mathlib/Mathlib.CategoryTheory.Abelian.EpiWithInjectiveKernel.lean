/-- The class of morphisms in an abelian category that are epimorphisms
and have an injective kernel. -/
def epiWithInjectiveKernel : MorphismProperty C :=
  fun _ _ f => Epi f ∧ Injective (kernel f)


/-- A morphism `g : X ⟶ Y` is epi with an injective kernel iff there exists a morphism
`f : I ⟶ X` with `I` injective such that `f ≫ g = 0` and
the short complex `I ⟶ X ⟶ Y` has a splitting. -/
lemma epiWithInjectiveKernel_iff {X Y : C} (g : X ⟶ Y) :
    epiWithInjectiveKernel g ↔ ∃ (I : C) (_ : Injective I) (f : I ⟶ X) (w : f ≫ g = 0),
      Nonempty (ShortComplex.mk f g w).Splitting := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    X Y : C
    g : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.Abelian.epiWithInjectiveKernel g) (Exists fun I => Exist …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Abelian C
      X Y : C
      g : Quiver.Hom X Y
      ⊢ CategoryTheory.Abelian.epiWithInjectiveKernel g → Exists fun I => Exists fun …
    -/
  · rintro ⟨_, _⟩
    /-
      case mp.intro
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Abelian C
      X Y : C
      g : Quiver.Hom X Y
      left✝ : CategoryTheory.Epi g
      right✝ : CategoryTheory.Injective (CategoryTheory.Limits.kernel g)
      ⊢ Exists fun I => Exists fun x => Exists fun f => Exists fun w => Nonempty (Ca …
    -/
    let S := ShortComplex.mk (kernel.ι g) g (by simp)
    exact ⟨_, inferInstance, _, S.zero,
      ⟨ShortComplex.Splitting.ofExactOfRetraction S
        (S.exact_of_f_is_kernel (kernelIsKernel g)) (Injective.factorThru (𝟙 _) (kernel.ι g))
        (by simp [S]) inferInstance⟩⟩
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Abelian C
      X Y : C
      g : Quiver.Hom X Y
      ⊢ (Exists fun I => Exists fun x => Exists fun f => Exists fun w => Nonempty (C …
    -/
  · rintro ⟨I, _,  f, w, ⟨σ⟩⟩
    /-
      case mpr.intro.intro.intro.intro.intro
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Abelian C
      X Y : C
      g : Quiver.Hom X Y
      I : C
      w✝ : CategoryTheory.Injective I
      f : Quiver.Hom I X
      w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
      σ : (CategoryTheory.ShortComplex.mk f g w).Splitting
      ⊢ CategoryTheory.Abelian.epiWithInjectiveKernel g
    -/
    have : IsSplitEpi g := ⟨σ.s, σ.s_g⟩
    let e : I ≅ kernel g :=
      IsLimit.conePointUniqueUpToIso σ.shortExact.fIsKernel (limit.isLimit _)
    /-
      case mpr.intro.intro.intro.intro.intro
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Abelian C
      X Y : C
      g : Quiver.Hom X Y
      I : C
      w✝ : CategoryTheory.Injective I
      f : Quiver.Hom I X
      w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
      σ : (CategoryTheory.ShortComplex.mk f g w).Splitting
      this : CategoryTheory.IsSplitEpi g
      e : CategoryTheory.Iso I (CategoryTheory.Limits.kernel g) := ⋯.fIsKernel.coneP …
      ⊢ CategoryTheory.Abelian.epiWithInjectiveKernel g
    -/
    exact ⟨inferInstance, Injective.of_iso e inferInstance⟩
    /-
      🎉 no goals
    -/


lemma epiWithInjectiveKernel_of_iso {X Y : C} (f : X ⟶ Y) [IsIso f] :
    epiWithInjectiveKernel f := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Abelian C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f
    ⊢ CategoryTheory.Abelian.epiWithInjectiveKernel f
  -/
  rw [epiWithInjectiveKernel_iff]
  exact ⟨0, inferInstance, 0, by simp,
    ⟨ShortComplex.Splitting.ofIsZeroOfIsIso _ (isZero_zero C) (by dsimp; infer_instance)⟩⟩


instance : (epiWithInjectiveKernel : MorphismProperty C).IsMultiplicative where
  id_mem _ := epiWithInjectiveKernel_of_iso _
  comp_mem {X Y Z} g₁ g₂ hg₁ hg₂ := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Abelian C
      X Y Z : C
      g₁ : Quiver.Hom X Y
      g₂ : Quiver.Hom Y Z
      hg₁ : CategoryTheory.Abelian.epiWithInjectiveKernel g₁
      hg₂ : CategoryTheory.Abelian.epiWithInjectiveKernel g₂
      ⊢ CategoryTheory.Abelian.epiWithInjectiveKernel (CategoryTheory.CategoryStruct …
    -/
    rw [epiWithInjectiveKernel_iff] at hg₁ hg₂ ⊢
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Abelian C
      X Y Z : C
      g₁ : Quiver.Hom X Y
      g₂ : Quiver.Hom Y Z
      hg₁ : Exists fun I => Exists fun x => Exists fun f => Exists fun w => Nonempty …
      hg₂ : Exists fun I => Exists fun x => Exists fun f => Exists fun w => Nonempty …
      ⊢ Exists fun I => Exists fun x => Exists fun f => Exists fun w => Nonempty (Ca …
    -/
    obtain ⟨I₁, _, f₁, w₁, ⟨σ₁⟩⟩ := hg₁
    /-
      case intro.intro.intro.intro.intro
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Abelian C
      X Y Z : C
      g₁ : Quiver.Hom X Y
      g₂ : Quiver.Hom Y Z
      hg₂ : Exists fun I => Exists fun x => Exists fun f => Exists fun w => Nonempty …
      I₁ : C
      w✝ : CategoryTheory.Injective I₁
      f₁ : Quiver.Hom I₁ X
      w₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ g₁) 0
      σ₁ : (CategoryTheory.ShortComplex.mk f₁ g₁ w₁).Splitting
      ⊢ Exists fun I => Exists fun x => Exists fun f => Exists fun w => Nonempty (Ca …
    -/
    obtain ⟨I₂, _, f₂, w₂, ⟨σ₂⟩⟩ := hg₂
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Abelian C
      X Y Z : C
      g₁ : Quiver.Hom X Y
      g₂ : Quiver.Hom Y Z
      I₁ : C
      w✝¹ : CategoryTheory.Injective I₁
      f₁ : Quiver.Hom I₁ X
      w₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ g₁) 0
      σ₁ : (CategoryTheory.ShortComplex.mk f₁ g₁ w₁).Splitting
      I₂ : C
      w✝ : CategoryTheory.Injective I₂
      f₂ : Quiver.Hom I₂ Y
      w₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ g₂) 0
      σ₂ : (CategoryTheory.ShortComplex.mk f₂ g₂ w₂).Splitting
      ⊢ Exists fun I => Exists fun x => Exists fun f => Exists fun w => Nonempty (Ca …
    -/
    refine ⟨I₁ ⊞ I₂, inferInstance, biprod.fst ≫ f₁ + biprod.snd ≫ f₂ ≫ σ₁.s, ?_, ⟨?_⟩⟩
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_1
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Abelian C
        X Y Z : C
        g₁ : Quiver.Hom X Y
        g₂ : Quiver.Hom Y Z
        I₁ : C
        w✝¹ : CategoryTheory.Injective I₁
        f₁ : Quiver.Hom I₁ X
        w₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ g₁) 0
        σ₁ : (CategoryTheory.ShortComplex.mk f₁ g₁ w₁).Splitting
        I₂ : C
        w✝ : CategoryTheory.Injective I₂
        f₂ : Quiver.Hom I₂ Y
        w₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ g₂) 0
        σ₂ : (CategoryTheory.ShortComplex.mk f₂ g₂ w₂).Splitting
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (HAdd.hAdd (CategoryTheory.CategorySt …
      -/
    · ext
        /-
          case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_1.h₀
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Abelian C
          X Y Z : C
          g₁ : Quiver.Hom X Y
          g₂ : Quiver.Hom Y Z
          I₁ : C
          w✝¹ : CategoryTheory.Injective I₁
          f₁ : Quiver.Hom I₁ X
          w₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ g₁) 0
          σ₁ : (CategoryTheory.ShortComplex.mk f₁ g₁ w₁).Splitting
          I₂ : C
          w✝ : CategoryTheory.Injective I₂
          f₂ : Quiver.Hom I₂ Y
          w₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ g₂) 0
          σ₂ : (CategoryTheory.ShortComplex.mk f₂ g₂ w₂).Splitting
          ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.inl (Cat …
        -/
      · simp [reassoc_of% w₁]
        /-
          🎉 no goals
        -/
        /-
          case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_1.h₁
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Abelian C
          X Y Z : C
          g₁ : Quiver.Hom X Y
          g₂ : Quiver.Hom Y Z
          I₁ : C
          w✝¹ : CategoryTheory.Injective I₁
          f₁ : Quiver.Hom I₁ X
          w₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ g₁) 0
          σ₁ : (CategoryTheory.ShortComplex.mk f₁ g₁ w₁).Splitting
          I₂ : C
          w✝ : CategoryTheory.Injective I₂
          f₂ : Quiver.Hom I₂ Y
          w₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ g₂) 0
          σ₂ : (CategoryTheory.ShortComplex.mk f₂ g₂ w₂).Splitting
          ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.inr (Cat …
        -/
      · simp [reassoc_of% σ₁.s_g, w₂]
        /-
          🎉 no goals
        -/
    · exact
        { r := σ₁.r ≫ biprod.inl + g₁ ≫ σ₂.r ≫ biprod.inr
          s := σ₂.s ≫ σ₁.s
          f_r := by
            ext
            · simp [σ₁.f_r]
            · simp [reassoc_of% w₁]
            · simp
            · simp [reassoc_of% σ₁.s_g, σ₂.f_r]
          s_g := by simp [reassoc_of% σ₁.s_g, σ₂.s_g]
          id := by
            dsimp
            have h := g₁ ≫= σ₂.id =≫ σ₁.s
            simp only [add_comp, assoc, comp_add, id_comp] at h
            rw [← σ₁.id, ← h]
            simp only [comp_add, add_comp, assoc, BinaryBicone.inl_fst_assoc,
              BinaryBicone.inr_fst_assoc, zero_comp, comp_zero, add_zero,
              BinaryBicone.inl_snd_assoc, BinaryBicone.inr_snd_assoc, zero_add]
            abel }


