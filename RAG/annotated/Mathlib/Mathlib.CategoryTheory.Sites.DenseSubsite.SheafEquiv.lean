include K in
lemma isIso_ranCounit_app_of_isDenseSubsite (Y : Sheaf J A) (U X) :
    IsIso ((yoneda.map ((G.op.ranCounit.app Y.val).app (op U))).app (op X)) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Category.{u_4, u_2} D
    G : CategoryTheory.Functor C D
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    A : Type w
    inst✝² : CategoryTheory.Category.{w', w} A
    inst✝¹ : ∀ (X : Opposite D), CategoryTheory.Limits.HasLimitsOfShape (CategoryT …
    inst✝ : CategoryTheory.Functor.IsDenseSubsite J K G
    Y : CategoryTheory.Sheaf J A
    U : C
    X : A
    ⊢ CategoryTheory.IsIso ((CategoryTheory.yoneda.map ((G.op.ranCounit.app Y.val) …
  -/
  rw [isIso_iff_bijective]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Category.{u_4, u_2} D
    G : CategoryTheory.Functor C D
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    A : Type w
    inst✝² : CategoryTheory.Category.{w', w} A
    inst✝¹ : ∀ (X : Opposite D), CategoryTheory.Limits.HasLimitsOfShape (CategoryT …
    inst✝ : CategoryTheory.Functor.IsDenseSubsite J K G
    Y : CategoryTheory.Sheaf J A
    U : C
    X : A
    ⊢ Function.Bijective ((CategoryTheory.yoneda.map ((G.op.ranCounit.app Y.val).a …
  -/
  constructor
    /-
      case left
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Category.{u_4, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type w
      inst✝² : CategoryTheory.Category.{w', w} A
      inst✝¹ : ∀ (X : Opposite D), CategoryTheory.Limits.HasLimitsOfShape (CategoryT …
      inst✝ : CategoryTheory.Functor.IsDenseSubsite J K G
      Y : CategoryTheory.Sheaf J A
      U : C
      X : A
      ⊢ Function.Injective ((CategoryTheory.yoneda.map ((G.op.ranCounit.app Y.val).a …
    -/
  · intro f₁ f₂ e
    /-
      case left
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Category.{u_4, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type w
      inst✝² : CategoryTheory.Category.{w', w} A
      inst✝¹ : ∀ (X : Opposite D), CategoryTheory.Limits.HasLimitsOfShape (CategoryT …
      inst✝ : CategoryTheory.Functor.IsDenseSubsite J K G
      Y : CategoryTheory.Sheaf J A
      U : C
      X : A
      f₁ f₂ : (CategoryTheory.yoneda.obj (((G.op.ran.comp ((CategoryTheory.whiskerin …
      e : Eq ((CategoryTheory.yoneda.map ((G.op.ranCounit.app Y.val).app { unop := U …
      ⊢ Eq f₁ f₂
    -/
    apply (isPointwiseRightKanExtensionRanCounit G.op Y.1 (.op (G.obj U))).hom_ext
    /-
      case left
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Category.{u_4, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type w
      inst✝² : CategoryTheory.Category.{w', w} A
      inst✝¹ : ∀ (X : Opposite D), CategoryTheory.Limits.HasLimitsOfShape (CategoryT …
      inst✝ : CategoryTheory.Functor.IsDenseSubsite J K G
      Y : CategoryTheory.Sheaf J A
      U : C
      X : A
      f₁ f₂ : (CategoryTheory.yoneda.obj (((G.op.ran.comp ((CategoryTheory.whiskerin …
      e : Eq ((CategoryTheory.yoneda.map ((G.op.ranCounit.app Y.val).app { unop := U …
      ⊢ ∀ (j : CategoryTheory.StructuredArrow { unop := G.obj U } G.op), Eq (Categor …
    -/
    rintro ⟨⟨⟨⟩⟩, ⟨W⟩, g⟩
    /-
      case left.mk.mk.unit.op
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Category.{u_4, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type w
      inst✝² : CategoryTheory.Category.{w', w} A
      inst✝¹ : ∀ (X : Opposite D), CategoryTheory.Limits.HasLimitsOfShape (CategoryT …
      inst✝ : CategoryTheory.Functor.IsDenseSubsite J K G
      Y : CategoryTheory.Sheaf J A
      U : C
      X : A
      f₁ f₂ : (CategoryTheory.yoneda.obj (((G.op.ran.comp ((CategoryTheory.whiskerin …
      e : Eq ((CategoryTheory.yoneda.map ((G.op.ranCounit.app Y.val).app { unop := U …
      W : C
      g : Quiver.Hom ((CategoryTheory.Functor.fromPUnit { unop := G.obj U }).obj { a …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f₁ (((CategoryTheory.Functor.RightExt …
    -/
    obtain ⟨g, rfl⟩ : ∃ g' : G.obj W ⟶ G.obj U, g = g'.op := ⟨g.unop, rfl⟩
    /-
      case left.mk.mk.unit.op.intro
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Category.{u_4, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type w
      inst✝² : CategoryTheory.Category.{w', w} A
      inst✝¹ : ∀ (X : Opposite D), CategoryTheory.Limits.HasLimitsOfShape (CategoryT …
      inst✝ : CategoryTheory.Functor.IsDenseSubsite J K G
      Y : CategoryTheory.Sheaf J A
      U : C
      X : A
      f₁ f₂ : (CategoryTheory.yoneda.obj (((G.op.ran.comp ((CategoryTheory.whiskerin …
      e : Eq ((CategoryTheory.yoneda.map ((G.op.ranCounit.app Y.val).app { unop := U …
      W : C
      g : Quiver.Hom (G.obj W) (G.obj U)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f₁ (((CategoryTheory.Functor.RightExt …
    -/
    apply (Y.2 X _ (IsDenseSubsite.imageSieve_mem J K G g)).isSeparatedFor.ext
    /-
      case left.mk.mk.unit.op.intro
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Category.{u_4, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type w
      inst✝² : CategoryTheory.Category.{w', w} A
      inst✝¹ : ∀ (X : Opposite D), CategoryTheory.Limits.HasLimitsOfShape (CategoryT …
      inst✝ : CategoryTheory.Functor.IsDenseSubsite J K G
      Y : CategoryTheory.Sheaf J A
      U : C
      X : A
      f₁ f₂ : (CategoryTheory.yoneda.obj (((G.op.ran.comp ((CategoryTheory.whiskerin …
      e : Eq ((CategoryTheory.yoneda.map ((G.op.ranCounit.app Y.val).app { unop := U …
      W : C
      g : Quiver.Hom (G.obj W) (G.obj U)
      ⊢ ∀ ⦃Y_1 : C⦄ ⦃f : Quiver.Hom Y_1 W⦄, (G.imageSieve g).arrows f → Eq ((Y.val.c …
    -/
    dsimp
    /-
      case left.mk.mk.unit.op.intro
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Category.{u_4, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type w
      inst✝² : CategoryTheory.Category.{w', w} A
      inst✝¹ : ∀ (X : Opposite D), CategoryTheory.Limits.HasLimitsOfShape (CategoryT …
      inst✝ : CategoryTheory.Functor.IsDenseSubsite J K G
      Y : CategoryTheory.Sheaf J A
      U : C
      X : A
      f₁ f₂ : (CategoryTheory.yoneda.obj (((G.op.ran.comp ((CategoryTheory.whiskerin …
      e : Eq ((CategoryTheory.yoneda.map ((G.op.ranCounit.app Y.val).app { unop := U …
      W : C
      g : Quiver.Hom (G.obj W) (G.obj U)
      ⊢ ∀ ⦃Y_1 : C⦄ ⦃f : Quiver.Hom Y_1 W⦄, (G.imageSieve g).arrows f → Eq (Category …
    -/
    rintro V iVW ⟨iVU, e'⟩
    /-
      case left.mk.mk.unit.op.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Category.{u_4, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type w
      inst✝² : CategoryTheory.Category.{w', w} A
      inst✝¹ : ∀ (X : Opposite D), CategoryTheory.Limits.HasLimitsOfShape (CategoryT …
      inst✝ : CategoryTheory.Functor.IsDenseSubsite J K G
      Y : CategoryTheory.Sheaf J A
      U : C
      X : A
      f₁ f₂ : (CategoryTheory.yoneda.obj (((G.op.ran.comp ((CategoryTheory.whiskerin …
      e : Eq ((CategoryTheory.yoneda.map ((G.op.ranCounit.app Y.val).app { unop := U …
      W : C
      g : Quiver.Hom (G.obj W) (G.obj U)
      V : C
      iVW : Quiver.Hom V W
      iVU : Quiver.Hom V U
      e' : Eq (G.map iVU) (CategoryTheory.CategoryStruct.comp (G.map iVW) g)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
    -/
    have := congr($e ≫ Y.1.map iVU.op)
    simp only [comp_obj, yoneda_map_app, Category.assoc, coyoneda_obj_obj, comp_map,
      coyoneda_obj_map, ← NatTrans.naturality, op_obj, op_map, Quiver.Hom.unop_op, ← map_comp_assoc,
      ← op_comp, ← e'] at this ⊢
    /-
      case left.mk.mk.unit.op.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Category.{u_4, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type w
      inst✝² : CategoryTheory.Category.{w', w} A
      inst✝¹ : ∀ (X : Opposite D), CategoryTheory.Limits.HasLimitsOfShape (CategoryT …
      inst✝ : CategoryTheory.Functor.IsDenseSubsite J K G
      Y : CategoryTheory.Sheaf J A
      U : C
      X : A
      f₁ f₂ : (CategoryTheory.yoneda.obj (((G.op.ran.comp ((CategoryTheory.whiskerin …
      e : Eq ((CategoryTheory.yoneda.map ((G.op.ranCounit.app Y.val).app { unop := U …
      W : C
      g : Quiver.Hom (G.obj W) (G.obj U)
      V : C
      iVW : Quiver.Hom V W
      iVU : Quiver.Hom V U
      e' : Eq (G.map iVU) (CategoryTheory.CategoryStruct.comp (G.map iVW) g)
      this : Eq (CategoryTheory.CategoryStruct.comp f₁ (CategoryTheory.CategoryStruc …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f₁ (CategoryTheory.CategoryStruct.com …
    -/
    erw [← NatTrans.naturality] at this
    /-
      case left.mk.mk.unit.op.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Category.{u_4, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type w
      inst✝² : CategoryTheory.Category.{w', w} A
      inst✝¹ : ∀ (X : Opposite D), CategoryTheory.Limits.HasLimitsOfShape (CategoryT …
      inst✝ : CategoryTheory.Functor.IsDenseSubsite J K G
      Y : CategoryTheory.Sheaf J A
      U : C
      X : A
      f₁ f₂ : (CategoryTheory.yoneda.obj (((G.op.ran.comp ((CategoryTheory.whiskerin …
      e : Eq ((CategoryTheory.yoneda.map ((G.op.ranCounit.app Y.val).app { unop := U …
      W : C
      g : Quiver.Hom (G.obj W) (G.obj U)
      V : C
      iVW : Quiver.Hom V W
      iVU : Quiver.Hom V U
      e' : Eq (G.map iVU) (CategoryTheory.CategoryStruct.comp (G.map iVW) g)
      this : Eq (CategoryTheory.CategoryStruct.comp f₁ (CategoryTheory.CategoryStruc …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f₁ (CategoryTheory.CategoryStruct.com …
    -/
    exact this
    /-
      🎉 no goals
    -/
    /-
      case right
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Category.{u_4, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type w
      inst✝² : CategoryTheory.Category.{w', w} A
      inst✝¹ : ∀ (X : Opposite D), CategoryTheory.Limits.HasLimitsOfShape (CategoryT …
      inst✝ : CategoryTheory.Functor.IsDenseSubsite J K G
      Y : CategoryTheory.Sheaf J A
      U : C
      X : A
      ⊢ Function.Surjective ((CategoryTheory.yoneda.map ((G.op.ranCounit.app Y.val). …
    -/
  · intro f
    /-
      case right
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Category.{u_4, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type w
      inst✝² : CategoryTheory.Category.{w', w} A
      inst✝¹ : ∀ (X : Opposite D), CategoryTheory.Limits.HasLimitsOfShape (CategoryT …
      inst✝ : CategoryTheory.Functor.IsDenseSubsite J K G
      Y : CategoryTheory.Sheaf J A
      U : C
      X : A
      f : (CategoryTheory.yoneda.obj (((CategoryTheory.Functor.id (CategoryTheory.Fu …
      ⊢ Exists fun a => Eq ((CategoryTheory.yoneda.map ((G.op.ranCounit.app Y.val).a …
    -/
    have (X Y Z) (f : X ⟶ Y) (g : G.obj Y ⟶ G.obj Z) (hf : G.imageSieve g f) : Exists _ := hf
    /-
      case right
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Category.{u_4, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type w
      inst✝² : CategoryTheory.Category.{w', w} A
      inst✝¹ : ∀ (X : Opposite D), CategoryTheory.Limits.HasLimitsOfShape (CategoryT …
      inst✝ : CategoryTheory.Functor.IsDenseSubsite J K G
      Y : CategoryTheory.Sheaf J A
      U : C
      X : A
      f : (CategoryTheory.yoneda.obj (((CategoryTheory.Functor.id (CategoryTheory.Fu …
      this : ∀ (X Y Z : C) (f : Quiver.Hom X Y) (g : Quiver.Hom (G.obj Y) (G.obj Z)) …
      ⊢ Exists fun a => Eq ((CategoryTheory.yoneda.map ((G.op.ranCounit.app Y.val).a …
    -/
    choose l hl using this
    let c : Limits.Cone (StructuredArrow.proj (op (G.obj U)) G.op ⋙ Y.val) := by
      refine ⟨X, ⟨fun g ↦ ?_, ?_⟩⟩
      · refine Y.2.amalgamate ⟨_, IsDenseSubsite.imageSieve_mem J K G g.hom.unop⟩
          (fun I ↦ f ≫ Y.1.map (l _ _ _ _ _ I.hf).op) fun I₁ I₂ r ↦ ?_
        apply (Y.2 X _ (IsDenseSubsite.equalizer_mem J K G (r.g₁ ≫ l _ _ _ _ _ I₁.hf)
          (r.g₂ ≫ l _ _ _ _ _ I₂.hf) ?_)).isSeparatedFor.ext fun V iUV (hiUV : _ = _) ↦ ?_
        · simp only [const_obj_obj, op_obj, map_comp, hl]
          simp only [← map_comp_assoc, r.w]
        · simp [← map_comp, ← op_comp, hiUV]
      · dsimp
        rintro ⟨⟨⟨⟩⟩, ⟨W₁⟩, g₁⟩ ⟨⟨⟨⟩⟩, ⟨W₂⟩, g₂⟩ ⟨⟨⟨⟨⟩⟩⟩, i, hi⟩
        dsimp at g₁ g₂ i hi
        -- See issue https://github.com/leanprover-community/mathlib4/pull/15781 for tracking performance regressions of `rintro` as here
        have h : g₂ = g₁ ≫ (G.map i.unop).op := by simpa only [Category.id_comp] using hi
        rcases h with ⟨rfl⟩
        have h : ∃ g' : G.obj W₁ ⟶ G.obj U, g₁ = g'.op := ⟨g₁.unop, rfl⟩
        rcases h with ⟨g, rfl⟩
        have h : ∃ i' : W₂ ⟶ W₁, i = i'.op := ⟨i.unop, rfl⟩
        rcases h with ⟨i, rfl⟩
        simp only [const_obj_obj, id_obj, comp_obj, StructuredArrow.proj_obj, const_obj_map, op_obj,
          unop_comp, Quiver.Hom.unop_op, Category.id_comp, comp_map, StructuredArrow.proj_map]
        apply Y.2.hom_ext ⟨_, IsDenseSubsite.imageSieve_mem J K G (G.map i ≫ g)⟩
        intro I
        simp only [Presheaf.IsSheaf.amalgamate_map, Category.assoc, ← Functor.map_comp, ← op_comp]
        let I' : GrothendieckTopology.Cover.Arrow ⟨_, IsDenseSubsite.imageSieve_mem J K G g⟩ :=
          ⟨_, I.f ≫ i, ⟨l _ _ _ _ _ I.hf, by simp [hl]⟩⟩
        refine Eq.trans ?_ (Y.2.amalgamate_map _ _ _ I').symm
        apply (Y.2 X _ (IsDenseSubsite.equalizer_mem J K G (l _ _ _ _ _ I.hf)
          (l _ _ _ _ _ I'.hf) (by simp [I', hl]))).isSeparatedFor.ext
            fun V iUV (hiUV : _ = _) ↦ ?_
        simp [I', ← Functor.map_comp, ← op_comp, hiUV]
    /-
      case right
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Category.{u_4, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type w
      inst✝² : CategoryTheory.Category.{w', w} A
      inst✝¹ : ∀ (X : Opposite D), CategoryTheory.Limits.HasLimitsOfShape (CategoryT …
      inst✝ : CategoryTheory.Functor.IsDenseSubsite J K G
      Y : CategoryTheory.Sheaf J A
      U : C
      X : A
      f : (CategoryTheory.yoneda.obj (((CategoryTheory.Functor.id (CategoryTheory.Fu …
      l : (X Y Z : C) → (f : Quiver.Hom X Y) → (g : Quiver.Hom (G.obj Y) (G.obj Z))  …
      hl : ∀ (X Y Z : C) (f : Quiver.Hom X Y) (g : Quiver.Hom (G.obj Y) (G.obj Z)) ( …
      c : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
      ⊢ Exists fun a => Eq ((CategoryTheory.yoneda.map ((G.op.ranCounit.app Y.val).a …
    -/
    refine ⟨(isPointwiseRightKanExtensionRanCounit G.op Y.1 (.op (G.obj U))).lift c, ?_⟩
      /-
        case right
        C : Type u_1
        D : Type u_2
        inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
        inst✝³ : CategoryTheory.Category.{u_4, u_2} D
        G : CategoryTheory.Functor C D
        J : CategoryTheory.GrothendieckTopology C
        K : CategoryTheory.GrothendieckTopology D
        A : Type w
        inst✝² : CategoryTheory.Category.{w', w} A
        inst✝¹ : ∀ (X : Opposite D), CategoryTheory.Limits.HasLimitsOfShape (CategoryT …
        inst✝ : CategoryTheory.Functor.IsDenseSubsite J K G
        Y : CategoryTheory.Sheaf J A
        U : C
        X : A
        f : (CategoryTheory.yoneda.obj (((CategoryTheory.Functor.id (CategoryTheory.Fu …
        l : (X Y Z : C) → (f : Quiver.Hom X Y) → (g : Quiver.Hom (G.obj Y) (G.obj Z))  …
        hl : ∀ (X Y Z : C) (f : Quiver.Hom X Y) (g : Quiver.Hom (G.obj Y) (G.obj Z)) ( …
        c : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
        ⊢ Eq ((CategoryTheory.yoneda.map ((G.op.ranCounit.app Y.val).app { unop := U } …
      -/
    · have := (isPointwiseRightKanExtensionRanCounit G.op Y.1 (.op (G.obj U))).fac c (.mk (𝟙 _))
      simp only [id_obj, comp_obj, StructuredArrow.proj_obj, StructuredArrow.mk_right,
        RightExtension.coneAt_pt, RightExtension.mk_left, RightExtension.coneAt_π_app,
        const_obj_obj, op_obj, StructuredArrow.mk_hom_eq_self, map_id, whiskeringLeft_obj_obj,
        RightExtension.mk_hom, Category.id_comp, StructuredArrow.mk_left, unop_id] at this
      /-
        case right
        C : Type u_1
        D : Type u_2
        inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
        inst✝³ : CategoryTheory.Category.{u_4, u_2} D
        G : CategoryTheory.Functor C D
        J : CategoryTheory.GrothendieckTopology C
        K : CategoryTheory.GrothendieckTopology D
        A : Type w
        inst✝² : CategoryTheory.Category.{w', w} A
        inst✝¹ : ∀ (X : Opposite D), CategoryTheory.Limits.HasLimitsOfShape (CategoryT …
        inst✝ : CategoryTheory.Functor.IsDenseSubsite J K G
        Y : CategoryTheory.Sheaf J A
        U : C
        X : A
        f : (CategoryTheory.yoneda.obj (((CategoryTheory.Functor.id (CategoryTheory.Fu …
        l : (X Y Z : C) → (f : Quiver.Hom X Y) → (g : Quiver.Hom (G.obj Y) (G.obj Z))  …
        hl : ∀ (X Y Z : C) (f : Quiver.Hom X Y) (g : Quiver.Hom (G.obj Y) (G.obj Z)) ( …
        c : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
        this : Eq (CategoryTheory.CategoryStruct.comp ((G.op.isPointwiseRightKanExtens …
        ⊢ Eq ((CategoryTheory.yoneda.map ((G.op.ranCounit.app Y.val).app { unop := U } …
      -/
      simp only [c, id_obj, yoneda_map_app, this]
      /-
        case right
        C : Type u_1
        D : Type u_2
        inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
        inst✝³ : CategoryTheory.Category.{u_4, u_2} D
        G : CategoryTheory.Functor C D
        J : CategoryTheory.GrothendieckTopology C
        K : CategoryTheory.GrothendieckTopology D
        A : Type w
        inst✝² : CategoryTheory.Category.{w', w} A
        inst✝¹ : ∀ (X : Opposite D), CategoryTheory.Limits.HasLimitsOfShape (CategoryT …
        inst✝ : CategoryTheory.Functor.IsDenseSubsite J K G
        Y : CategoryTheory.Sheaf J A
        U : C
        X : A
        f : (CategoryTheory.yoneda.obj (((CategoryTheory.Functor.id (CategoryTheory.Fu …
        l : (X Y Z : C) → (f : Quiver.Hom X Y) → (g : Quiver.Hom (G.obj Y) (G.obj Z))  …
        hl : ∀ (X Y Z : C) (f : Quiver.Hom X Y) (g : Quiver.Hom (G.obj Y) (G.obj Z)) ( …
        c : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
        this : Eq (CategoryTheory.CategoryStruct.comp ((G.op.isPointwiseRightKanExtens …
        ⊢ Eq (⋯.amalgamate ⟨G.imageSieve (CategoryTheory.CategoryStruct.id (G.obj U)), …
      -/
      apply Y.2.hom_ext ⟨_, IsDenseSubsite.imageSieve_mem J K G (𝟙 (G.obj U))⟩ _ _ fun I ↦ ?_
      apply (Y.2 X _ (IsDenseSubsite.equalizer_mem J K G (l _ _ _ _ _ I.hf)
        I.f (by simp [hl]))).isSeparatedFor.ext fun V iUV (hiUV : _ = _) ↦ ?_
      /-
        C : Type u_1
        D : Type u_2
        inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
        inst✝³ : CategoryTheory.Category.{u_4, u_2} D
        G : CategoryTheory.Functor C D
        J : CategoryTheory.GrothendieckTopology C
        K : CategoryTheory.GrothendieckTopology D
        A : Type w
        inst✝² : CategoryTheory.Category.{w', w} A
        inst✝¹ : ∀ (X : Opposite D), CategoryTheory.Limits.HasLimitsOfShape (CategoryT …
        inst✝ : CategoryTheory.Functor.IsDenseSubsite J K G
        Y : CategoryTheory.Sheaf J A
        U : C
        X : A
        f : (CategoryTheory.yoneda.obj (((CategoryTheory.Functor.id (CategoryTheory.Fu …
        l : (X Y Z : C) → (f : Quiver.Hom X Y) → (g : Quiver.Hom (G.obj Y) (G.obj Z))  …
        hl : ∀ (X Y Z : C) (f : Quiver.Hom X Y) (g : Quiver.Hom (G.obj Y) (G.obj Z)) ( …
        c : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
        this : Eq (CategoryTheory.CategoryStruct.comp ((G.op.isPointwiseRightKanExtens …
        I : CategoryTheory.GrothendieckTopology.Cover.Arrow ⟨G.imageSieve (CategoryThe …
        V : C
        iUV : Quiver.Hom V I.Y
        hiUV : Eq (CategoryTheory.CategoryStruct.comp iUV (l I.Y U U I.f (CategoryTheo …
        ⊢ Eq ((Y.val.comp (CategoryTheory.coyoneda.obj { unop := X })).map iUV.op (Cat …
      -/
      simp [← Functor.map_comp, ← op_comp, hiUV]
      /-
        🎉 no goals
      -/


instance (Y : Sheaf J A) : IsIso ((G.sheafAdjunctionCocontinuous A J K).counit.app Y) := by
  apply (config := { allowSynthFailures := true })
    ReflectsIsomorphisms.reflects (sheafToPresheaf J A)
  /-
    case inst
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Category.{u_4, u_2} D
    G : CategoryTheory.Functor C D
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    A : Type w
    inst✝² : CategoryTheory.Category.{w', w} A
    inst✝¹ : ∀ (X : Opposite D), CategoryTheory.Limits.HasLimitsOfShape (CategoryT …
    inst✝ : CategoryTheory.Functor.IsDenseSubsite J K G
    Y : CategoryTheory.Sheaf J A
    ⊢ CategoryTheory.IsIso ((CategoryTheory.sheafToPresheaf J A).map ((G.sheafAdju …
  -/
  rw [NatTrans.isIso_iff_isIso_app]
  /-
    case inst
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Category.{u_4, u_2} D
    G : CategoryTheory.Functor C D
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    A : Type w
    inst✝² : CategoryTheory.Category.{w', w} A
    inst✝¹ : ∀ (X : Opposite D), CategoryTheory.Limits.HasLimitsOfShape (CategoryT …
    inst✝ : CategoryTheory.Functor.IsDenseSubsite J K G
    Y : CategoryTheory.Sheaf J A
    ⊢ ∀ (X : Opposite C), CategoryTheory.IsIso (((CategoryTheory.sheafToPresheaf J …
  -/
  intro ⟨U⟩
  /-
    case inst
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Category.{u_4, u_2} D
    G : CategoryTheory.Functor C D
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    A : Type w
    inst✝² : CategoryTheory.Category.{w', w} A
    inst✝¹ : ∀ (X : Opposite D), CategoryTheory.Limits.HasLimitsOfShape (CategoryT …
    inst✝ : CategoryTheory.Functor.IsDenseSubsite J K G
    Y : CategoryTheory.Sheaf J A
    U : C
    ⊢ CategoryTheory.IsIso (((CategoryTheory.sheafToPresheaf J A).map ((G.sheafAdj …
  -/
  apply (config := { allowSynthFailures := true }) ReflectsIsomorphisms.reflects yoneda
  /-
    case inst
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Category.{u_4, u_2} D
    G : CategoryTheory.Functor C D
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    A : Type w
    inst✝² : CategoryTheory.Category.{w', w} A
    inst✝¹ : ∀ (X : Opposite D), CategoryTheory.Limits.HasLimitsOfShape (CategoryT …
    inst✝ : CategoryTheory.Functor.IsDenseSubsite J K G
    Y : CategoryTheory.Sheaf J A
    U : C
    ⊢ CategoryTheory.IsIso (CategoryTheory.yoneda.map (((CategoryTheory.sheafToPre …
  -/
  rw [NatTrans.isIso_iff_isIso_app]
  /-
    case inst
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Category.{u_4, u_2} D
    G : CategoryTheory.Functor C D
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    A : Type w
    inst✝² : CategoryTheory.Category.{w', w} A
    inst✝¹ : ∀ (X : Opposite D), CategoryTheory.Limits.HasLimitsOfShape (CategoryT …
    inst✝ : CategoryTheory.Functor.IsDenseSubsite J K G
    Y : CategoryTheory.Sheaf J A
    U : C
    ⊢ ∀ (X : Opposite A), CategoryTheory.IsIso ((CategoryTheory.yoneda.map (((Cate …
  -/
  intro ⟨X⟩
  simp only [comp_obj, sheafToPresheaf_obj, sheafPushforwardContinuous_obj_val_obj, yoneda_obj_obj,
    id_obj, sheafToPresheaf_map, sheafAdjunctionCocontinuous_counit_app_val, ranAdjunction_counit]
  /-
    case inst
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Category.{u_4, u_2} D
    G : CategoryTheory.Functor C D
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    A : Type w
    inst✝² : CategoryTheory.Category.{w', w} A
    inst✝¹ : ∀ (X : Opposite D), CategoryTheory.Limits.HasLimitsOfShape (CategoryT …
    inst✝ : CategoryTheory.Functor.IsDenseSubsite J K G
    Y : CategoryTheory.Sheaf J A
    U : C
    X : A
    ⊢ CategoryTheory.IsIso ((CategoryTheory.yoneda.map ((G.op.ranCounit.app Y.val) …
  -/
  exact isIso_ranCounit_app_of_isDenseSubsite G J K Y U X
  /-
    🎉 no goals
  -/


/--
If `G : C ⥤ D` exhibits `(C, J)` as a dense subsite of `(D, K)`,
it induces an equivalence of category of sheaves valued in a category with suitable limits.
-/
@[simps! functor inverse]
noncomputable def sheafEquiv : Sheaf J A ≌ Sheaf K A :=
  (G.sheafAdjunctionCocontinuous A J K).toEquivalence.symm


instance : (G.sheafPushforwardContinuous A J K).IsEquivalence :=
  inferInstanceAs (IsDenseSubsite.sheafEquiv G _ _ _).inverse.IsEquivalence


/-- The natural isomorphism exhibiting the compatibility of
`IsDenseSubsite.sheafEquiv` with sheafification. -/
noncomputable
abbrev sheafEquivSheafificationCompatibility :
    (whiskeringLeft _ _ A).obj G.op ⋙ presheafToSheaf _ _ ≅
      presheafToSheaf _ _ ⋙ (sheafEquiv G J K A).inverse := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{?u.150987, u_1} C
    inst✝⁵ : CategoryTheory.Category.{?u.150991, u_2} D
    G : CategoryTheory.Functor C D
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    A : Type w
    inst✝⁴ : CategoryTheory.Category.{w', w} A
    inst✝³ : ∀ (X : Opposite D), CategoryTheory.Limits.HasLimitsOfShape (CategoryT …
    inst✝² : CategoryTheory.Functor.IsDenseSubsite J K G
    inst✝¹ : CategoryTheory.HasWeakSheafify J A
    inst✝ : CategoryTheory.HasWeakSheafify K A
    ⊢ CategoryTheory.Iso (((CategoryTheory.whiskeringLeft (Opposite C) (Opposite D …
  -/
  apply Functor.pushforwardContinuousSheafificationCompatibility
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-23")]
alias IsCoverDense.sheafEquivOfCoverPreservingCoverLifting := IsDenseSubsite.sheafEquiv

@[deprecated (since := "2024-07-23")]
alias IsCoverDense.sheafEquivOfCoverPreservingCoverLiftingSheafificationCompatibility :=
  IsDenseSubsite.sheafEquivSheafificationCompatibility


