/-- We say that `𝒢` is a separating set if the functors `C(G, -)` for `G ∈ 𝒢` are collectively
    faithful, i.e., if `h ≫ f = h ≫ g` for all `h` with domain in `𝒢` implies `f = g`. -/
def IsSeparating (𝒢 : Set C) : Prop :=
  ∀ ⦃X Y : C⦄ (f g : X ⟶ Y), (∀ G ∈ 𝒢, ∀ (h : G ⟶ X), h ≫ f = h ≫ g) → f = g


/-- We say that `𝒢` is a coseparating set if the functors `C(-, G)` for `G ∈ 𝒢` are collectively
    faithful, i.e., if `f ≫ h = g ≫ h` for all `h` with codomain in `𝒢` implies `f = g`. -/
def IsCoseparating (𝒢 : Set C) : Prop :=
  ∀ ⦃X Y : C⦄ (f g : X ⟶ Y), (∀ G ∈ 𝒢, ∀ (h : Y ⟶ G), f ≫ h = g ≫ h) → f = g


/-- We say that `𝒢` is a detecting set if the functors `C(G, -)` collectively reflect isomorphisms,
    i.e., if any `h` with domain in `𝒢` uniquely factors through `f`, then `f` is an isomorphism. -/
def IsDetecting (𝒢 : Set C) : Prop :=
  ∀ ⦃X Y : C⦄ (f : X ⟶ Y), (∀ G ∈ 𝒢, ∀ (h : G ⟶ Y), ∃! h' : G ⟶ X, h' ≫ f = h) → IsIso f


/-- We say that `𝒢` is a codetecting set if the functors `C(-, G)` collectively reflect
    isomorphisms, i.e., if any `h` with codomain in `G` uniquely factors through `f`, then `f` is
    an isomorphism. -/
def IsCodetecting (𝒢 : Set C) : Prop :=
  ∀ ⦃X Y : C⦄ (f : X ⟶ Y), (∀ G ∈ 𝒢, ∀ (h : X ⟶ G), ∃! h' : Y ⟶ G, f ≫ h' = h) → IsIso f


lemma IsSeparating.of_equivalence
    {𝒢 : Set C} (h : IsSeparating 𝒢) {D : Type*} [Category D] (α : C ≌ D) :
    IsSeparating (α.functor.obj '' 𝒢) := fun X Y f g H =>
  α.inverse.map_injective (h _ _ (fun Z hZ h => by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      𝒢 : Set C
      h✝ : CategoryTheory.IsSeparating 𝒢
      D : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} D
      α : CategoryTheory.Equivalence C D
      X Y : D
      f g : Quiver.Hom X Y
      H : ∀ (G : D), Membership.mem (Set.image α.functor.obj 𝒢) G → ∀ (h : Quiver.Ho …
      Z : C
      hZ : Membership.mem 𝒢 Z
      h : Quiver.Hom Z (α.inverse.obj X)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp h (α.inverse.map f)) (CategoryTheory. …
    -/
    obtain ⟨h', rfl⟩ := (α.toAdjunction.homEquiv _ _).surjective h
    simp only [Adjunction.homEquiv_unit, Category.assoc, ← Functor.map_comp,
      H (α.functor.obj Z) (Set.mem_image_of_mem _ hZ) h']))


lemma IsCoseparating.of_equivalence
    {𝒢 : Set C} (h : IsCoseparating 𝒢) {D : Type*} [Category D] (α : C ≌ D) :
    IsCoseparating (α.functor.obj '' 𝒢) := fun X Y f g H =>
  α.inverse.map_injective (h _ _ (fun Z hZ h => by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      𝒢 : Set C
      h✝ : CategoryTheory.IsCoseparating 𝒢
      D : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} D
      α : CategoryTheory.Equivalence C D
      X Y : D
      f g : Quiver.Hom X Y
      H : ∀ (G : D), Membership.mem (Set.image α.functor.obj 𝒢) G → ∀ (h : Quiver.Ho …
      Z : C
      hZ : Membership.mem 𝒢 Z
      h : Quiver.Hom (α.inverse.obj Y) Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (α.inverse.map f) h) (CategoryTheory. …
    -/
    obtain ⟨h', rfl⟩ := (α.symm.toAdjunction.homEquiv _ _).symm.surjective h
    simp only [Adjunction.homEquiv_symm_apply, ← Category.assoc, ← Functor.map_comp,
      Equivalence.symm_functor, H (α.functor.obj Z) (Set.mem_image_of_mem _ hZ) h']))


theorem isSeparating_op_iff (𝒢 : Set C) : IsSeparating 𝒢.op ↔ IsCoseparating 𝒢 := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    𝒢 : Set C
    ⊢ Iff (CategoryTheory.IsSeparating 𝒢.op) (CategoryTheory.IsCoseparating 𝒢)
  -/
  refine ⟨fun h𝒢 X Y f g hfg => ?_, fun h𝒢 X Y f g hfg => ?_⟩
    /-
      case refine_1
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      𝒢 : Set C
      h𝒢 : CategoryTheory.IsSeparating 𝒢.op
      X Y : C
      f g : Quiver.Hom X Y
      hfg : ∀ (G : C), Membership.mem 𝒢 G → ∀ (h : Quiver.Hom Y G), Eq (CategoryTheo …
      ⊢ Eq f g
    -/
  · refine Quiver.Hom.op_inj (h𝒢 _ _ fun G hG h => Quiver.Hom.unop_inj ?_)
    /-
      case refine_1
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      𝒢 : Set C
      h𝒢 : CategoryTheory.IsSeparating 𝒢.op
      X Y : C
      f g : Quiver.Hom X Y
      hfg : ∀ (G : C), Membership.mem 𝒢 G → ∀ (h : Quiver.Hom Y G), Eq (CategoryTheo …
      G : Opposite C
      hG : Membership.mem 𝒢.op G
      h : Quiver.Hom G { unop := Y }
      ⊢ Eq (CategoryTheory.CategoryStruct.comp h f.op).unop (CategoryTheory.Category …
    -/
    simpa only [unop_comp, Quiver.Hom.unop_op] using hfg _ (Set.mem_op.1 hG) _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      𝒢 : Set C
      h𝒢 : CategoryTheory.IsCoseparating 𝒢
      X Y : Opposite C
      f g : Quiver.Hom X Y
      hfg : ∀ (G : Opposite C), Membership.mem 𝒢.op G → ∀ (h : Quiver.Hom G X), Eq ( …
      ⊢ Eq f g
    -/
  · refine Quiver.Hom.unop_inj (h𝒢 _ _ fun G hG h => Quiver.Hom.op_inj ?_)
    /-
      case refine_2
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      𝒢 : Set C
      h𝒢 : CategoryTheory.IsCoseparating 𝒢
      X Y : Opposite C
      f g : Quiver.Hom X Y
      hfg : ∀ (G : Opposite C), Membership.mem 𝒢.op G → ∀ (h : Quiver.Hom G X), Eq ( …
      G : C
      hG : Membership.mem 𝒢 G
      h : Quiver.Hom (Opposite.unop X) G
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f.unop h).op (CategoryTheory.Category …
    -/
    simpa only [op_comp, Quiver.Hom.op_unop] using hfg _ (Set.op_mem_op.2 hG) _
    /-
      🎉 no goals
    -/


theorem isCoseparating_op_iff (𝒢 : Set C) : IsCoseparating 𝒢.op ↔ IsSeparating 𝒢 := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    𝒢 : Set C
    ⊢ Iff (CategoryTheory.IsCoseparating 𝒢.op) (CategoryTheory.IsSeparating 𝒢)
  -/
  refine ⟨fun h𝒢 X Y f g hfg => ?_, fun h𝒢 X Y f g hfg => ?_⟩
    /-
      case refine_1
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      𝒢 : Set C
      h𝒢 : CategoryTheory.IsCoseparating 𝒢.op
      X Y : C
      f g : Quiver.Hom X Y
      hfg : ∀ (G : C), Membership.mem 𝒢 G → ∀ (h : Quiver.Hom G X), Eq (CategoryTheo …
      ⊢ Eq f g
    -/
  · refine Quiver.Hom.op_inj (h𝒢 _ _ fun G hG h => Quiver.Hom.unop_inj ?_)
    /-
      case refine_1
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      𝒢 : Set C
      h𝒢 : CategoryTheory.IsCoseparating 𝒢.op
      X Y : C
      f g : Quiver.Hom X Y
      hfg : ∀ (G : C), Membership.mem 𝒢 G → ∀ (h : Quiver.Hom G X), Eq (CategoryTheo …
      G : Opposite C
      hG : Membership.mem 𝒢.op G
      h : Quiver.Hom { unop := X } G
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f.op h).unop (CategoryTheory.Category …
    -/
    simpa only [unop_comp, Quiver.Hom.unop_op] using hfg _ (Set.mem_op.1 hG) _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      𝒢 : Set C
      h𝒢 : CategoryTheory.IsSeparating 𝒢
      X Y : Opposite C
      f g : Quiver.Hom X Y
      hfg : ∀ (G : Opposite C), Membership.mem 𝒢.op G → ∀ (h : Quiver.Hom Y G), Eq ( …
      ⊢ Eq f g
    -/
  · refine Quiver.Hom.unop_inj (h𝒢 _ _ fun G hG h => Quiver.Hom.op_inj ?_)
    /-
      case refine_2
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      𝒢 : Set C
      h𝒢 : CategoryTheory.IsSeparating 𝒢
      X Y : Opposite C
      f g : Quiver.Hom X Y
      hfg : ∀ (G : Opposite C), Membership.mem 𝒢.op G → ∀ (h : Quiver.Hom Y G), Eq ( …
      G : C
      hG : Membership.mem 𝒢 G
      h : Quiver.Hom G (Opposite.unop Y)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp h f.unop).op (CategoryTheory.Category …
    -/
    simpa only [op_comp, Quiver.Hom.op_unop] using hfg _ (Set.op_mem_op.2 hG) _
    /-
      🎉 no goals
    -/


theorem isCoseparating_unop_iff (𝒢 : Set Cᵒᵖ) : IsCoseparating 𝒢.unop ↔ IsSeparating 𝒢 := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    𝒢 : Set (Opposite C)
    ⊢ Iff (CategoryTheory.IsCoseparating 𝒢.unop) (CategoryTheory.IsSeparating 𝒢)
  -/
  rw [← isSeparating_op_iff, Set.unop_op]
  /-
    🎉 no goals
  -/


theorem isSeparating_unop_iff (𝒢 : Set Cᵒᵖ) : IsSeparating 𝒢.unop ↔ IsCoseparating 𝒢 := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    𝒢 : Set (Opposite C)
    ⊢ Iff (CategoryTheory.IsSeparating 𝒢.unop) (CategoryTheory.IsCoseparating 𝒢)
  -/
  rw [← isCoseparating_op_iff, Set.unop_op]
  /-
    🎉 no goals
  -/


theorem isDetecting_op_iff (𝒢 : Set C) : IsDetecting 𝒢.op ↔ IsCodetecting 𝒢 := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    𝒢 : Set C
    ⊢ Iff (CategoryTheory.IsDetecting 𝒢.op) (CategoryTheory.IsCodetecting 𝒢)
  -/
  refine ⟨fun h𝒢 X Y f hf => ?_, fun h𝒢 X Y f hf => ?_⟩
    /-
      case refine_1
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      𝒢 : Set C
      h𝒢 : CategoryTheory.IsDetecting 𝒢.op
      X Y : C
      f : Quiver.Hom X Y
      hf : ∀ (G : C), Membership.mem 𝒢 G → ∀ (h : Quiver.Hom X G), ExistsUnique fun  …
      ⊢ CategoryTheory.IsIso f
    -/
  · refine (isIso_op_iff _).1 (h𝒢 _ fun G hG h => ?_)
    /-
      case refine_1
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      𝒢 : Set C
      h𝒢 : CategoryTheory.IsDetecting 𝒢.op
      X Y : C
      f : Quiver.Hom X Y
      hf : ∀ (G : C), Membership.mem 𝒢 G → ∀ (h : Quiver.Hom X G), ExistsUnique fun  …
      G : Opposite C
      hG : Membership.mem 𝒢.op G
      h : Quiver.Hom G { unop := X }
      ⊢ ExistsUnique fun h' => Eq (CategoryTheory.CategoryStruct.comp h' f.op) h
    -/
    obtain ⟨t, ht, ht'⟩ := hf (unop G) (Set.mem_op.1 hG) h.unop
    exact
      ⟨t.op, Quiver.Hom.unop_inj ht, fun y hy => Quiver.Hom.unop_inj (ht' _ (Quiver.Hom.op_inj hy))⟩
    /-
      case refine_2
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      𝒢 : Set C
      h𝒢 : CategoryTheory.IsCodetecting 𝒢
      X Y : Opposite C
      f : Quiver.Hom X Y
      hf : ∀ (G : Opposite C), Membership.mem 𝒢.op G → ∀ (h : Quiver.Hom G Y), Exist …
      ⊢ CategoryTheory.IsIso f
    -/
  · refine (isIso_unop_iff _).1 (h𝒢 _ fun G hG h => ?_)
    /-
      case refine_2
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      𝒢 : Set C
      h𝒢 : CategoryTheory.IsCodetecting 𝒢
      X Y : Opposite C
      f : Quiver.Hom X Y
      hf : ∀ (G : Opposite C), Membership.mem 𝒢.op G → ∀ (h : Quiver.Hom G Y), Exist …
      G : C
      hG : Membership.mem 𝒢 G
      h : Quiver.Hom (Opposite.unop Y) G
      ⊢ ExistsUnique fun h' => Eq (CategoryTheory.CategoryStruct.comp f.unop h') h
    -/
    obtain ⟨t, ht, ht'⟩ := hf (op G) (Set.op_mem_op.2 hG) h.op
    /-
      case refine_2.intro.intro
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      𝒢 : Set C
      h𝒢 : CategoryTheory.IsCodetecting 𝒢
      X Y : Opposite C
      f : Quiver.Hom X Y
      hf : ∀ (G : Opposite C), Membership.mem 𝒢.op G → ∀ (h : Quiver.Hom G Y), Exist …
      G : C
      hG : Membership.mem 𝒢 G
      h : Quiver.Hom (Opposite.unop Y) G
      t : Quiver.Hom { unop := G } X
      ht : Eq (CategoryTheory.CategoryStruct.comp t f) h.op
      ht' : ∀ (y : Quiver.Hom { unop := G } X), (fun h' => Eq (CategoryTheory.Catego …
      ⊢ ExistsUnique fun h' => Eq (CategoryTheory.CategoryStruct.comp f.unop h') h
    -/
    refine ⟨t.unop, Quiver.Hom.op_inj ht, fun y hy => Quiver.Hom.op_inj (ht' _ ?_)⟩
    /-
      case refine_2.intro.intro
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      𝒢 : Set C
      h𝒢 : CategoryTheory.IsCodetecting 𝒢
      X Y : Opposite C
      f : Quiver.Hom X Y
      hf : ∀ (G : Opposite C), Membership.mem 𝒢.op G → ∀ (h : Quiver.Hom G Y), Exist …
      G : C
      hG : Membership.mem 𝒢 G
      h : Quiver.Hom (Opposite.unop Y) G
      t : Quiver.Hom { unop := G } X
      ht : Eq (CategoryTheory.CategoryStruct.comp t f) h.op
      ht' : ∀ (y : Quiver.Hom { unop := G } X), (fun h' => Eq (CategoryTheory.Catego …
      y : Quiver.Hom (Opposite.unop X) G
      hy : (fun h' => Eq (CategoryTheory.CategoryStruct.comp f.unop h') h) y
      ⊢ (fun h' => Eq (CategoryTheory.CategoryStruct.comp h' f) h.op) y.op
    -/
    exact Quiver.Hom.unop_inj (by simpa only using hy)
    /-
      🎉 no goals
    -/


theorem isCodetecting_op_iff (𝒢 : Set C) : IsCodetecting 𝒢.op ↔ IsDetecting 𝒢 := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    𝒢 : Set C
    ⊢ Iff (CategoryTheory.IsCodetecting 𝒢.op) (CategoryTheory.IsDetecting 𝒢)
  -/
  refine ⟨fun h𝒢 X Y f hf => ?_, fun h𝒢 X Y f hf => ?_⟩
    /-
      case refine_1
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      𝒢 : Set C
      h𝒢 : CategoryTheory.IsCodetecting 𝒢.op
      X Y : C
      f : Quiver.Hom X Y
      hf : ∀ (G : C), Membership.mem 𝒢 G → ∀ (h : Quiver.Hom G Y), ExistsUnique fun  …
      ⊢ CategoryTheory.IsIso f
    -/
  · refine (isIso_op_iff _).1 (h𝒢 _ fun G hG h => ?_)
    /-
      case refine_1
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      𝒢 : Set C
      h𝒢 : CategoryTheory.IsCodetecting 𝒢.op
      X Y : C
      f : Quiver.Hom X Y
      hf : ∀ (G : C), Membership.mem 𝒢 G → ∀ (h : Quiver.Hom G Y), ExistsUnique fun  …
      G : Opposite C
      hG : Membership.mem 𝒢.op G
      h : Quiver.Hom { unop := Y } G
      ⊢ ExistsUnique fun h' => Eq (CategoryTheory.CategoryStruct.comp f.op h') h
    -/
    obtain ⟨t, ht, ht'⟩ := hf (unop G) (Set.mem_op.1 hG) h.unop
    exact
      ⟨t.op, Quiver.Hom.unop_inj ht, fun y hy => Quiver.Hom.unop_inj (ht' _ (Quiver.Hom.op_inj hy))⟩
    /-
      case refine_2
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      𝒢 : Set C
      h𝒢 : CategoryTheory.IsDetecting 𝒢
      X Y : Opposite C
      f : Quiver.Hom X Y
      hf : ∀ (G : Opposite C), Membership.mem 𝒢.op G → ∀ (h : Quiver.Hom X G), Exist …
      ⊢ CategoryTheory.IsIso f
    -/
  · refine (isIso_unop_iff _).1 (h𝒢 _ fun G hG h => ?_)
    /-
      case refine_2
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      𝒢 : Set C
      h𝒢 : CategoryTheory.IsDetecting 𝒢
      X Y : Opposite C
      f : Quiver.Hom X Y
      hf : ∀ (G : Opposite C), Membership.mem 𝒢.op G → ∀ (h : Quiver.Hom X G), Exist …
      G : C
      hG : Membership.mem 𝒢 G
      h : Quiver.Hom G (Opposite.unop X)
      ⊢ ExistsUnique fun h' => Eq (CategoryTheory.CategoryStruct.comp h' f.unop) h
    -/
    obtain ⟨t, ht, ht'⟩ := hf (op G) (Set.op_mem_op.2 hG) h.op
    /-
      case refine_2.intro.intro
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      𝒢 : Set C
      h𝒢 : CategoryTheory.IsDetecting 𝒢
      X Y : Opposite C
      f : Quiver.Hom X Y
      hf : ∀ (G : Opposite C), Membership.mem 𝒢.op G → ∀ (h : Quiver.Hom X G), Exist …
      G : C
      hG : Membership.mem 𝒢 G
      h : Quiver.Hom G (Opposite.unop X)
      t : Quiver.Hom Y { unop := G }
      ht : Eq (CategoryTheory.CategoryStruct.comp f t) h.op
      ht' : ∀ (y : Quiver.Hom Y { unop := G }), (fun h' => Eq (CategoryTheory.Catego …
      ⊢ ExistsUnique fun h' => Eq (CategoryTheory.CategoryStruct.comp h' f.unop) h
    -/
    refine ⟨t.unop, Quiver.Hom.op_inj ht, fun y hy => Quiver.Hom.op_inj (ht' _ ?_)⟩
    /-
      case refine_2.intro.intro
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      𝒢 : Set C
      h𝒢 : CategoryTheory.IsDetecting 𝒢
      X Y : Opposite C
      f : Quiver.Hom X Y
      hf : ∀ (G : Opposite C), Membership.mem 𝒢.op G → ∀ (h : Quiver.Hom X G), Exist …
      G : C
      hG : Membership.mem 𝒢 G
      h : Quiver.Hom G (Opposite.unop X)
      t : Quiver.Hom Y { unop := G }
      ht : Eq (CategoryTheory.CategoryStruct.comp f t) h.op
      ht' : ∀ (y : Quiver.Hom Y { unop := G }), (fun h' => Eq (CategoryTheory.Catego …
      y : Quiver.Hom G (Opposite.unop Y)
      hy : (fun h' => Eq (CategoryTheory.CategoryStruct.comp h' f.unop) h) y
      ⊢ (fun h' => Eq (CategoryTheory.CategoryStruct.comp f h') h.op) y.op
    -/
    exact Quiver.Hom.unop_inj (by simpa only using hy)
    /-
      🎉 no goals
    -/


theorem isDetecting_unop_iff (𝒢 : Set Cᵒᵖ) : IsDetecting 𝒢.unop ↔ IsCodetecting 𝒢 := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    𝒢 : Set (Opposite C)
    ⊢ Iff (CategoryTheory.IsDetecting 𝒢.unop) (CategoryTheory.IsCodetecting 𝒢)
  -/
  rw [← isCodetecting_op_iff, Set.unop_op]
  /-
    🎉 no goals
  -/


theorem isCodetecting_unop_iff {𝒢 : Set Cᵒᵖ} : IsCodetecting 𝒢.unop ↔ IsDetecting 𝒢 := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    𝒢 : Set (Opposite C)
    ⊢ Iff (CategoryTheory.IsCodetecting 𝒢.unop) (CategoryTheory.IsDetecting 𝒢)
  -/
  rw [← isDetecting_op_iff, Set.unop_op]
  /-
    🎉 no goals
  -/


theorem IsDetecting.isSeparating [HasEqualizers C] {𝒢 : Set C} (h𝒢 : IsDetecting 𝒢) :
    IsSeparating 𝒢 := fun _ _ f g hfg =>
  have : IsIso (equalizer.ι f g) := h𝒢 _ fun _ hG _ => equalizer.existsUnique _ (hfg _ hG _)
  eq_of_epi_equalizer


theorem IsCodetecting.isCoseparating [HasCoequalizers C] {𝒢 : Set C} :
    IsCodetecting 𝒢 → IsCoseparating 𝒢 := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Limits.HasCoequalizers C
    𝒢 : Set C
    ⊢ CategoryTheory.IsCodetecting 𝒢 → CategoryTheory.IsCoseparating 𝒢
  -/
  simpa only [← isSeparating_op_iff, ← isDetecting_op_iff] using IsDetecting.isSeparating
  /-
    🎉 no goals
  -/


theorem IsSeparating.isDetecting [Balanced C] {𝒢 : Set C} (h𝒢 : IsSeparating 𝒢) :
    IsDetecting 𝒢 := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Balanced C
    𝒢 : Set C
    h𝒢 : CategoryTheory.IsSeparating 𝒢
    ⊢ CategoryTheory.IsDetecting 𝒢
  -/
  intro X Y f hf
  refine
    (isIso_iff_mono_and_epi _).2 ⟨⟨fun g h hgh => h𝒢 _ _ fun G hG i => ?_⟩, ⟨fun g h hgh => ?_⟩⟩
    /-
      case refine_1
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Balanced C
      𝒢 : Set C
      h𝒢 : CategoryTheory.IsSeparating 𝒢
      X Y : C
      f : Quiver.Hom X Y
      hf : ∀ (G : C), Membership.mem 𝒢 G → ∀ (h : Quiver.Hom G Y), ExistsUnique fun  …
      Z✝ : C
      g h : Quiver.Hom Z✝ X
      hgh : Eq (CategoryTheory.CategoryStruct.comp g f) (CategoryTheory.CategoryStru …
      G : C
      hG : Membership.mem 𝒢 G
      i : Quiver.Hom G Z✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp i g) (CategoryTheory.CategoryStruct.c …
    -/
  · obtain ⟨t, -, ht⟩ := hf G hG (i ≫ g ≫ f)
    /-
      case refine_1.intro.intro
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Balanced C
      𝒢 : Set C
      h𝒢 : CategoryTheory.IsSeparating 𝒢
      X Y : C
      f : Quiver.Hom X Y
      hf : ∀ (G : C), Membership.mem 𝒢 G → ∀ (h : Quiver.Hom G Y), ExistsUnique fun  …
      Z✝ : C
      g h : Quiver.Hom Z✝ X
      hgh : Eq (CategoryTheory.CategoryStruct.comp g f) (CategoryTheory.CategoryStru …
      G : C
      hG : Membership.mem 𝒢 G
      i : Quiver.Hom G Z✝
      t : Quiver.Hom G X
      ht : ∀ (y : Quiver.Hom G X), (fun h' => Eq (CategoryTheory.CategoryStruct.comp …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp i g) (CategoryTheory.CategoryStruct.c …
    -/
    rw [ht (i ≫ g) (Category.assoc _ _ _), ht (i ≫ h) (hgh.symm ▸ Category.assoc _ _ _)]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Balanced C
      𝒢 : Set C
      h𝒢 : CategoryTheory.IsSeparating 𝒢
      X Y : C
      f : Quiver.Hom X Y
      hf : ∀ (G : C), Membership.mem 𝒢 G → ∀ (h : Quiver.Hom G Y), ExistsUnique fun  …
      Z✝ : C
      g h : Quiver.Hom Y Z✝
      hgh : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStru …
      ⊢ Eq g h
    -/
  · refine h𝒢 _ _ fun G hG i => ?_
    /-
      case refine_2
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Balanced C
      𝒢 : Set C
      h𝒢 : CategoryTheory.IsSeparating 𝒢
      X Y : C
      f : Quiver.Hom X Y
      hf : ∀ (G : C), Membership.mem 𝒢 G → ∀ (h : Quiver.Hom G Y), ExistsUnique fun  …
      Z✝ : C
      g h : Quiver.Hom Y Z✝
      hgh : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStru …
      G : C
      hG : Membership.mem 𝒢 G
      i : Quiver.Hom G Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp i g) (CategoryTheory.CategoryStruct.c …
    -/
    obtain ⟨t, rfl, -⟩ := hf G hG i
    /-
      case refine_2.intro.intro
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Balanced C
      𝒢 : Set C
      h𝒢 : CategoryTheory.IsSeparating 𝒢
      X Y : C
      f : Quiver.Hom X Y
      hf : ∀ (G : C), Membership.mem 𝒢 G → ∀ (h : Quiver.Hom G Y), ExistsUnique fun  …
      Z✝ : C
      g h : Quiver.Hom Y Z✝
      hgh : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStru …
      G : C
      hG : Membership.mem 𝒢 G
      t : Quiver.Hom G X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp t …
    -/
    rw [Category.assoc, hgh, Category.assoc]
    /-
      🎉 no goals
    -/


theorem IsCoseparating.isCodetecting [Balanced C] {𝒢 : Set C} :
    IsCoseparating 𝒢 → IsCodetecting 𝒢 := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Balanced C
    𝒢 : Set C
    ⊢ CategoryTheory.IsCoseparating 𝒢 → CategoryTheory.IsCodetecting 𝒢
  -/
  simpa only [← isDetecting_op_iff, ← isSeparating_op_iff] using IsSeparating.isDetecting
  /-
    🎉 no goals
  -/


theorem isDetecting_iff_isSeparating [HasEqualizers C] [Balanced C] (𝒢 : Set C) :
    IsDetecting 𝒢 ↔ IsSeparating 𝒢 :=
  ⟨IsDetecting.isSeparating, IsSeparating.isDetecting⟩


theorem isCodetecting_iff_isCoseparating [HasCoequalizers C] [Balanced C] {𝒢 : Set C} :
    IsCodetecting 𝒢 ↔ IsCoseparating 𝒢 :=
  ⟨IsCodetecting.isCoseparating, IsCoseparating.isCodetecting⟩


theorem IsSeparating.mono {𝒢 : Set C} (h𝒢 : IsSeparating 𝒢) {ℋ : Set C} (h𝒢ℋ : 𝒢 ⊆ ℋ) :
    IsSeparating ℋ := fun _ _ _ _ hfg => h𝒢 _ _ fun _ hG _ => hfg _ (h𝒢ℋ hG) _


theorem IsCoseparating.mono {𝒢 : Set C} (h𝒢 : IsCoseparating 𝒢) {ℋ : Set C} (h𝒢ℋ : 𝒢 ⊆ ℋ) :
    IsCoseparating ℋ := fun _ _ _ _ hfg => h𝒢 _ _ fun _ hG _ => hfg _ (h𝒢ℋ hG) _


theorem IsDetecting.mono {𝒢 : Set C} (h𝒢 : IsDetecting 𝒢) {ℋ : Set C} (h𝒢ℋ : 𝒢 ⊆ ℋ) :
    IsDetecting ℋ := fun _ _ _ hf => h𝒢 _ fun _ hG _ => hf _ (h𝒢ℋ hG) _


theorem IsCodetecting.mono {𝒢 : Set C} (h𝒢 : IsCodetecting 𝒢) {ℋ : Set C} (h𝒢ℋ : 𝒢 ⊆ ℋ) :
    IsCodetecting ℋ := fun _ _ _ hf => h𝒢 _ fun _ hG _ => hf _ (h𝒢ℋ hG) _


theorem thin_of_isSeparating_empty (h : IsSeparating (∅ : Set C)) : Quiver.IsThin C := fun _ _ =>
  ⟨fun _ _ => h _ _ fun _ => False.elim⟩


theorem isSeparating_empty_of_thin [Quiver.IsThin C] : IsSeparating (∅ : Set C) :=
  fun _ _ _ _ _ => Subsingleton.elim _ _


theorem thin_of_isCoseparating_empty (h : IsCoseparating (∅ : Set C)) : Quiver.IsThin C :=
  fun _ _ => ⟨fun _ _ => h _ _ fun _ => False.elim⟩


theorem isCoseparating_empty_of_thin [Quiver.IsThin C] : IsCoseparating (∅ : Set C) :=
  fun _ _ _ _ _ => Subsingleton.elim _ _


theorem groupoid_of_isDetecting_empty (h : IsDetecting (∅ : Set C)) {X Y : C} (f : X ⟶ Y) :
    IsIso f :=
  h _ fun _ => False.elim


theorem isDetecting_empty_of_groupoid [∀ {X Y : C} (f : X ⟶ Y), IsIso f] :
    IsDetecting (∅ : Set C) := fun _ _ _ _ => inferInstance


theorem groupoid_of_isCodetecting_empty (h : IsCodetecting (∅ : Set C)) {X Y : C} (f : X ⟶ Y) :
    IsIso f :=
  h _ fun _ => False.elim


theorem isCodetecting_empty_of_groupoid [∀ {X Y : C} (f : X ⟶ Y), IsIso f] :
    IsCodetecting (∅ : Set C) := fun _ _ _ _ => inferInstance


theorem isSeparating_iff_epi (𝒢 : Set C)
    [∀ A : C, HasCoproduct fun f : ΣG : 𝒢, (G : C) ⟶ A => (f.1 : C)] :
    IsSeparating 𝒢 ↔ ∀ A : C, Epi (Sigma.desc (@Sigma.snd 𝒢 fun G => (G : C) ⟶ A)) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    𝒢 : Set C
    inst✝ : ∀ (A : C), CategoryTheory.Limits.HasCoproduct fun f => ↑f.fst
    ⊢ Iff (CategoryTheory.IsSeparating 𝒢) (∀ (A : C), CategoryTheory.Epi (Category …
  -/
  refine ⟨fun h A => ⟨fun u v huv => h _ _ fun G hG f => ?_⟩, fun h X Y f g hh => ?_⟩
    /-
      case refine_1
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      𝒢 : Set C
      inst✝ : ∀ (A : C), CategoryTheory.Limits.HasCoproduct fun f => ↑f.fst
      h : CategoryTheory.IsSeparating 𝒢
      A Z✝ : C
      u v : Quiver.Hom A Z✝
      huv : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.desc …
      G : C
      hG : Membership.mem 𝒢 G
      f : Quiver.Hom G A
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f u) (CategoryTheory.CategoryStruct.c …
    -/
  · simpa using Sigma.ι (fun f : ΣG : 𝒢, (G : C) ⟶ A => (f.1 : C)) ⟨⟨G, hG⟩, f⟩ ≫= huv
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      𝒢 : Set C
      inst✝ : ∀ (A : C), CategoryTheory.Limits.HasCoproduct fun f => ↑f.fst
      h : ∀ (A : C), CategoryTheory.Epi (CategoryTheory.Limits.Sigma.desc Sigma.snd)
      X Y : C
      f g : Quiver.Hom X Y
      hh : ∀ (G : C), Membership.mem 𝒢 G → ∀ (h : Quiver.Hom G X), Eq (CategoryTheor …
      ⊢ Eq f g
    -/
  · haveI := h X
    refine
      (cancel_epi (Sigma.desc (@Sigma.snd 𝒢 fun G => (G : C) ⟶ X))).1 (colimit.hom_ext fun j => ?_)
    /-
      case refine_2
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      𝒢 : Set C
      inst✝ : ∀ (A : C), CategoryTheory.Limits.HasCoproduct fun f => ↑f.fst
      h : ∀ (A : C), CategoryTheory.Epi (CategoryTheory.Limits.Sigma.desc Sigma.snd)
      X Y : C
      f g : Quiver.Hom X Y
      hh : ∀ (G : C), Membership.mem 𝒢 G → ∀ (h : Quiver.Hom G X), Eq (CategoryTheor …
      this : CategoryTheory.Epi (CategoryTheory.Limits.Sigma.desc Sigma.snd)
      j : CategoryTheory.Discrete (Sigma fun G => Quiver.Hom (↑G) X)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (Cat …
    -/
    simpa using hh j.as.1.1 j.as.1.2 j.as.2
    /-
      🎉 no goals
    -/


theorem isCoseparating_iff_mono (𝒢 : Set C)
    [∀ A : C, HasProduct fun f : ΣG : 𝒢, A ⟶ (G : C) => (f.1 : C)] :
    IsCoseparating 𝒢 ↔ ∀ A : C, Mono (Pi.lift (@Sigma.snd 𝒢 fun G => A ⟶ (G : C))) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    𝒢 : Set C
    inst✝ : ∀ (A : C), CategoryTheory.Limits.HasProduct fun f => ↑f.fst
    ⊢ Iff (CategoryTheory.IsCoseparating 𝒢) (∀ (A : C), CategoryTheory.Mono (Categ …
  -/
  refine ⟨fun h A => ⟨fun u v huv => h _ _ fun G hG f => ?_⟩, fun h X Y f g hh => ?_⟩
    /-
      case refine_1
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      𝒢 : Set C
      inst✝ : ∀ (A : C), CategoryTheory.Limits.HasProduct fun f => ↑f.fst
      h : CategoryTheory.IsCoseparating 𝒢
      A Z✝ : C
      u v : Quiver.Hom Z✝ A
      huv : Eq (CategoryTheory.CategoryStruct.comp u (CategoryTheory.Limits.Pi.lift  …
      G : C
      hG : Membership.mem 𝒢 G
      f : Quiver.Hom A G
      ⊢ Eq (CategoryTheory.CategoryStruct.comp u f) (CategoryTheory.CategoryStruct.c …
    -/
  · simpa using huv =≫ Pi.π (fun f : ΣG : 𝒢, A ⟶ (G : C) => (f.1 : C)) ⟨⟨G, hG⟩, f⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      𝒢 : Set C
      inst✝ : ∀ (A : C), CategoryTheory.Limits.HasProduct fun f => ↑f.fst
      h : ∀ (A : C), CategoryTheory.Mono (CategoryTheory.Limits.Pi.lift Sigma.snd)
      X Y : C
      f g : Quiver.Hom X Y
      hh : ∀ (G : C), Membership.mem 𝒢 G → ∀ (h : Quiver.Hom Y G), Eq (CategoryTheor …
      ⊢ Eq f g
    -/
  · haveI := h Y
    /-
      case refine_2
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      𝒢 : Set C
      inst✝ : ∀ (A : C), CategoryTheory.Limits.HasProduct fun f => ↑f.fst
      h : ∀ (A : C), CategoryTheory.Mono (CategoryTheory.Limits.Pi.lift Sigma.snd)
      X Y : C
      f g : Quiver.Hom X Y
      hh : ∀ (G : C), Membership.mem 𝒢 G → ∀ (h : Quiver.Hom Y G), Eq (CategoryTheor …
      this : CategoryTheory.Mono (CategoryTheory.Limits.Pi.lift Sigma.snd)
      ⊢ Eq f g
    -/
    refine (cancel_mono (Pi.lift (@Sigma.snd 𝒢 fun G => Y ⟶ (G : C)))).1 (limit.hom_ext fun j => ?_)
    /-
      case refine_2
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      𝒢 : Set C
      inst✝ : ∀ (A : C), CategoryTheory.Limits.HasProduct fun f => ↑f.fst
      h : ∀ (A : C), CategoryTheory.Mono (CategoryTheory.Limits.Pi.lift Sigma.snd)
      X Y : C
      f g : Quiver.Hom X Y
      hh : ∀ (G : C), Membership.mem 𝒢 G → ∀ (h : Quiver.Hom Y G), Eq (CategoryTheor …
      this : CategoryTheory.Mono (CategoryTheory.Limits.Pi.lift Sigma.snd)
      j : CategoryTheory.Discrete (Sigma fun G => Quiver.Hom Y ↑G)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
    -/
    simpa using hh j.as.1.1 j.as.1.2 j.as.2
    /-
      🎉 no goals
    -/


/-- An ingredient of the proof of the Special Adjoint Functor Theorem: a complete well-powered
    category with a small coseparating set has an initial object.

    In fact, it follows from the Special Adjoint Functor Theorem that `C` is already cocomplete,
    see `hasColimits_of_hasLimits_of_isCoseparating`. -/
theorem hasInitial_of_isCoseparating [LocallySmall.{w} C] [WellPowered.{w} C]
    [HasLimitsOfSize.{w, w} C] {𝒢 : Set C} [Small.{w} 𝒢]
    (h𝒢 : IsCoseparating 𝒢) : HasInitial C := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.LocallySmall.{w, v₁, u₁} C
    inst✝² : CategoryTheory.WellPowered.{w, v₁, u₁} C
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfSize.{w, w, v₁, u₁} C
    𝒢 : Set C
    inst✝ : Small.{w, u₁} ↑𝒢
    h𝒢 : CategoryTheory.IsCoseparating 𝒢
    ⊢ CategoryTheory.Limits.HasInitial C
  -/
  have := hasFiniteLimits_of_hasLimitsOfSize C
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.LocallySmall.{w, v₁, u₁} C
    inst✝² : CategoryTheory.WellPowered.{w, v₁, u₁} C
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfSize.{w, w, v₁, u₁} C
    𝒢 : Set C
    inst✝ : Small.{w, u₁} ↑𝒢
    h𝒢 : CategoryTheory.IsCoseparating 𝒢
    this : CategoryTheory.Limits.HasFiniteLimits C
    ⊢ CategoryTheory.Limits.HasInitial C
  -/
  haveI : HasProductsOfShape 𝒢 C := hasProductsOfShape_of_small C 𝒢
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.LocallySmall.{w, v₁, u₁} C
    inst✝² : CategoryTheory.WellPowered.{w, v₁, u₁} C
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfSize.{w, w, v₁, u₁} C
    𝒢 : Set C
    inst✝ : Small.{w, u₁} ↑𝒢
    h𝒢 : CategoryTheory.IsCoseparating 𝒢
    this✝ : CategoryTheory.Limits.HasFiniteLimits C
    this : CategoryTheory.Limits.HasProductsOfShape (↑𝒢) C
    ⊢ CategoryTheory.Limits.HasInitial C
  -/
  haveI := fun A => hasProductsOfShape_of_small.{w} C (ΣG : 𝒢, A ⟶ (G : C))
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.LocallySmall.{w, v₁, u₁} C
    inst✝² : CategoryTheory.WellPowered.{w, v₁, u₁} C
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfSize.{w, w, v₁, u₁} C
    𝒢 : Set C
    inst✝ : Small.{w, u₁} ↑𝒢
    h𝒢 : CategoryTheory.IsCoseparating 𝒢
    this✝¹ : CategoryTheory.Limits.HasFiniteLimits C
    this✝ : CategoryTheory.Limits.HasProductsOfShape (↑𝒢) C
    this : ∀ (A : C), CategoryTheory.Limits.HasProductsOfShape (Sigma fun G => Qui …
    ⊢ CategoryTheory.Limits.HasInitial C
  -/
  letI := completeLatticeOfCompleteSemilatticeInf (Subobject (piObj (Subtype.val : 𝒢 → C)))
  suffices ∀ A : C, Unique (((⊥ : Subobject (piObj (Subtype.val : 𝒢 → C))) : C) ⟶ A) by
    exact hasInitial_of_unique ((⊥ : Subobject (piObj (Subtype.val : 𝒢 → C))) : C)
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.LocallySmall.{w, v₁, u₁} C
    inst✝² : CategoryTheory.WellPowered.{w, v₁, u₁} C
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfSize.{w, w, v₁, u₁} C
    𝒢 : Set C
    inst✝ : Small.{w, u₁} ↑𝒢
    h𝒢 : CategoryTheory.IsCoseparating 𝒢
    this✝² : CategoryTheory.Limits.HasFiniteLimits C
    this✝¹ : CategoryTheory.Limits.HasProductsOfShape (↑𝒢) C
    this✝ : ∀ (A : C), CategoryTheory.Limits.HasProductsOfShape (Sigma fun G => Qu …
    this : CompleteLattice (CategoryTheory.Subobject (CategoryTheory.Limits.piObj  …
    ⊢ (A : C) → Unique (Quiver.Hom (CategoryTheory.Subobject.underlying.obj Bot.bo …
  -/
  refine fun A => ⟨⟨?_⟩, fun f => ?_⟩
    /-
      case refine_1
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.LocallySmall.{w, v₁, u₁} C
      inst✝² : CategoryTheory.WellPowered.{w, v₁, u₁} C
      inst✝¹ : CategoryTheory.Limits.HasLimitsOfSize.{w, w, v₁, u₁} C
      𝒢 : Set C
      inst✝ : Small.{w, u₁} ↑𝒢
      h𝒢 : CategoryTheory.IsCoseparating 𝒢
      this✝² : CategoryTheory.Limits.HasFiniteLimits C
      this✝¹ : CategoryTheory.Limits.HasProductsOfShape (↑𝒢) C
      this✝ : ∀ (A : C), CategoryTheory.Limits.HasProductsOfShape (Sigma fun G => Qu …
      this : CompleteLattice (CategoryTheory.Subobject (CategoryTheory.Limits.piObj  …
      A : C
      ⊢ Quiver.Hom (CategoryTheory.Subobject.underlying.obj Bot.bot) A
    -/
  · let s := Pi.lift fun f : ΣG : 𝒢, A ⟶ (G : C) => id (Pi.π (Subtype.val : 𝒢 → C)) f.1
    /-
      case refine_1
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.LocallySmall.{w, v₁, u₁} C
      inst✝² : CategoryTheory.WellPowered.{w, v₁, u₁} C
      inst✝¹ : CategoryTheory.Limits.HasLimitsOfSize.{w, w, v₁, u₁} C
      𝒢 : Set C
      inst✝ : Small.{w, u₁} ↑𝒢
      h𝒢 : CategoryTheory.IsCoseparating 𝒢
      this✝² : CategoryTheory.Limits.HasFiniteLimits C
      this✝¹ : CategoryTheory.Limits.HasProductsOfShape (↑𝒢) C
      this✝ : ∀ (A : C), CategoryTheory.Limits.HasProductsOfShape (Sigma fun G => Qu …
      this : CompleteLattice (CategoryTheory.Subobject (CategoryTheory.Limits.piObj  …
      A : C
      s : Quiver.Hom (CategoryTheory.Limits.piObj Subtype.val) (CategoryTheory.Limit …
      ⊢ Quiver.Hom (CategoryTheory.Subobject.underlying.obj Bot.bot) A
    -/
    let t := Pi.lift (@Sigma.snd 𝒢 fun G => A ⟶ (G : C))
    /-
      case refine_1
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.LocallySmall.{w, v₁, u₁} C
      inst✝² : CategoryTheory.WellPowered.{w, v₁, u₁} C
      inst✝¹ : CategoryTheory.Limits.HasLimitsOfSize.{w, w, v₁, u₁} C
      𝒢 : Set C
      inst✝ : Small.{w, u₁} ↑𝒢
      h𝒢 : CategoryTheory.IsCoseparating 𝒢
      this✝² : CategoryTheory.Limits.HasFiniteLimits C
      this✝¹ : CategoryTheory.Limits.HasProductsOfShape (↑𝒢) C
      this✝ : ∀ (A : C), CategoryTheory.Limits.HasProductsOfShape (Sigma fun G => Qu …
      this : CompleteLattice (CategoryTheory.Subobject (CategoryTheory.Limits.piObj  …
      A : C
      s : Quiver.Hom (CategoryTheory.Limits.piObj Subtype.val) (CategoryTheory.Limit …
      t : Quiver.Hom A (CategoryTheory.Limits.piObj fun b => ↑b.fst) := CategoryTheo …
      ⊢ Quiver.Hom (CategoryTheory.Subobject.underlying.obj Bot.bot) A
    -/
    haveI : Mono t := (isCoseparating_iff_mono 𝒢).1 h𝒢 A
    /-
      case refine_1
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.LocallySmall.{w, v₁, u₁} C
      inst✝² : CategoryTheory.WellPowered.{w, v₁, u₁} C
      inst✝¹ : CategoryTheory.Limits.HasLimitsOfSize.{w, w, v₁, u₁} C
      𝒢 : Set C
      inst✝ : Small.{w, u₁} ↑𝒢
      h𝒢 : CategoryTheory.IsCoseparating 𝒢
      this✝³ : CategoryTheory.Limits.HasFiniteLimits C
      this✝² : CategoryTheory.Limits.HasProductsOfShape (↑𝒢) C
      this✝¹ : ∀ (A : C), CategoryTheory.Limits.HasProductsOfShape (Sigma fun G => Q …
      this✝ : CompleteLattice (CategoryTheory.Subobject (CategoryTheory.Limits.piObj …
      A : C
      s : Quiver.Hom (CategoryTheory.Limits.piObj Subtype.val) (CategoryTheory.Limit …
      t : Quiver.Hom A (CategoryTheory.Limits.piObj fun b => ↑b.fst) := CategoryTheo …
      this : CategoryTheory.Mono t
      ⊢ Quiver.Hom (CategoryTheory.Subobject.underlying.obj Bot.bot) A
    -/
    exact Subobject.ofLEMk _ (pullback.fst _ _ : pullback s t ⟶ _) bot_le ≫ pullback.snd _ _
    /-
      🎉 no goals
    -/
  · suffices ∀ (g : Subobject.underlying.obj ⊥ ⟶ A), f = g by
      apply this
    /-
      case refine_2
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.LocallySmall.{w, v₁, u₁} C
      inst✝² : CategoryTheory.WellPowered.{w, v₁, u₁} C
      inst✝¹ : CategoryTheory.Limits.HasLimitsOfSize.{w, w, v₁, u₁} C
      𝒢 : Set C
      inst✝ : Small.{w, u₁} ↑𝒢
      h𝒢 : CategoryTheory.IsCoseparating 𝒢
      this✝² : CategoryTheory.Limits.HasFiniteLimits C
      this✝¹ : CategoryTheory.Limits.HasProductsOfShape (↑𝒢) C
      this✝ : ∀ (A : C), CategoryTheory.Limits.HasProductsOfShape (Sigma fun G => Qu …
      this : CompleteLattice (CategoryTheory.Subobject (CategoryTheory.Limits.piObj  …
      A : C
      f : Quiver.Hom (CategoryTheory.Subobject.underlying.obj Bot.bot) A
      ⊢ ∀ (g : Quiver.Hom (CategoryTheory.Subobject.underlying.obj Bot.bot) A), Eq f g
    -/
    intro g
    /-
      case refine_2
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.LocallySmall.{w, v₁, u₁} C
      inst✝² : CategoryTheory.WellPowered.{w, v₁, u₁} C
      inst✝¹ : CategoryTheory.Limits.HasLimitsOfSize.{w, w, v₁, u₁} C
      𝒢 : Set C
      inst✝ : Small.{w, u₁} ↑𝒢
      h𝒢 : CategoryTheory.IsCoseparating 𝒢
      this✝² : CategoryTheory.Limits.HasFiniteLimits C
      this✝¹ : CategoryTheory.Limits.HasProductsOfShape (↑𝒢) C
      this✝ : ∀ (A : C), CategoryTheory.Limits.HasProductsOfShape (Sigma fun G => Qu …
      this : CompleteLattice (CategoryTheory.Subobject (CategoryTheory.Limits.piObj  …
      A : C
      f : Quiver.Hom (CategoryTheory.Subobject.underlying.obj Bot.bot) A
      g : Quiver.Hom (CategoryTheory.Subobject.underlying.obj Bot.bot) A
      ⊢ Eq f g
    -/
    suffices IsSplitEpi (equalizer.ι f g) by exact eq_of_epi_equalizer
    exact IsSplitEpi.mk' ⟨Subobject.ofLEMk _ (equalizer.ι f g ≫ Subobject.arrow _) bot_le, by
      ext
      simp⟩


/-- An ingredient of the proof of the Special Adjoint Functor Theorem: a cocomplete well-copowered
    category with a small separating set has a terminal object.

    In fact, it follows from the Special Adjoint Functor Theorem that `C` is already complete, see
    `hasLimits_of_hasColimits_of_isSeparating`. -/
theorem hasTerminal_of_isSeparating [LocallySmall.{w} Cᵒᵖ] [WellPowered.{w} Cᵒᵖ]
    [HasColimitsOfSize.{w, w} C] {𝒢 : Set C} [Small.{w} 𝒢]
    (h𝒢 : IsSeparating 𝒢) : HasTerminal C := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.LocallySmall.{w, v₁, u₁} (Opposite C)
    inst✝² : CategoryTheory.WellPowered.{w, v₁, u₁} (Opposite C)
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfSize.{w, w, v₁, u₁} C
    𝒢 : Set C
    inst✝ : Small.{w, u₁} ↑𝒢
    h𝒢 : CategoryTheory.IsSeparating 𝒢
    ⊢ CategoryTheory.Limits.HasTerminal C
  -/
  haveI : Small.{w} 𝒢.op := small_of_injective (Set.opEquiv_self 𝒢).injective
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.LocallySmall.{w, v₁, u₁} (Opposite C)
    inst✝² : CategoryTheory.WellPowered.{w, v₁, u₁} (Opposite C)
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfSize.{w, w, v₁, u₁} C
    𝒢 : Set C
    inst✝ : Small.{w, u₁} ↑𝒢
    h𝒢 : CategoryTheory.IsSeparating 𝒢
    this : Small.{w, u₁} ↑𝒢.op
    ⊢ CategoryTheory.Limits.HasTerminal C
  -/
  haveI : HasInitial Cᵒᵖ := hasInitial_of_isCoseparating ((isCoseparating_op_iff _).2 h𝒢)
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.LocallySmall.{w, v₁, u₁} (Opposite C)
    inst✝² : CategoryTheory.WellPowered.{w, v₁, u₁} (Opposite C)
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfSize.{w, w, v₁, u₁} C
    𝒢 : Set C
    inst✝ : Small.{w, u₁} ↑𝒢
    h𝒢 : CategoryTheory.IsSeparating 𝒢
    this✝ : Small.{w, u₁} ↑𝒢.op
    this : CategoryTheory.Limits.HasInitial (Opposite C)
    ⊢ CategoryTheory.Limits.HasTerminal C
  -/
  exact hasTerminal_of_hasInitial_op
  /-
    🎉 no goals
  -/


theorem eq_of_le_of_isDetecting {𝒢 : Set C} (h𝒢 : IsDetecting 𝒢) {X : C} (P Q : Subobject X)
    (h₁ : P ≤ Q) (h₂ : ∀ G ∈ 𝒢, ∀ {f : G ⟶ X}, Q.Factors f → P.Factors f) : P = Q := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    𝒢 : Set C
    h𝒢 : CategoryTheory.IsDetecting 𝒢
    X : C
    P Q : CategoryTheory.Subobject X
    h₁ : LE.le P Q
    h₂ : ∀ (G : C), Membership.mem 𝒢 G → ∀ {f : Quiver.Hom G X}, Q.Factors f → P.F …
    ⊢ Eq P Q
  -/
  suffices IsIso (ofLE _ _ h₁) by exact le_antisymm h₁ (le_of_comm (inv (ofLE _ _ h₁)) (by simp))
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    𝒢 : Set C
    h𝒢 : CategoryTheory.IsDetecting 𝒢
    X : C
    P Q : CategoryTheory.Subobject X
    h₁ : LE.le P Q
    h₂ : ∀ (G : C), Membership.mem 𝒢 G → ∀ {f : Quiver.Hom G X}, Q.Factors f → P.F …
    ⊢ CategoryTheory.IsIso (P.ofLE Q h₁)
  -/
  refine h𝒢 _ fun G hG f => ?_
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    𝒢 : Set C
    h𝒢 : CategoryTheory.IsDetecting 𝒢
    X : C
    P Q : CategoryTheory.Subobject X
    h₁ : LE.le P Q
    h₂ : ∀ (G : C), Membership.mem 𝒢 G → ∀ {f : Quiver.Hom G X}, Q.Factors f → P.F …
    G : C
    hG : Membership.mem 𝒢 G
    f : Quiver.Hom G (CategoryTheory.Subobject.underlying.obj Q)
    ⊢ ExistsUnique fun h' => Eq (CategoryTheory.CategoryStruct.comp h' (P.ofLE Q h …
  -/
  have : P.Factors (f ≫ Q.arrow) := h₂ _ hG ((factors_iff _ _).2 ⟨_, rfl⟩)
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    𝒢 : Set C
    h𝒢 : CategoryTheory.IsDetecting 𝒢
    X : C
    P Q : CategoryTheory.Subobject X
    h₁ : LE.le P Q
    h₂ : ∀ (G : C), Membership.mem 𝒢 G → ∀ {f : Quiver.Hom G X}, Q.Factors f → P.F …
    G : C
    hG : Membership.mem 𝒢 G
    f : Quiver.Hom G (CategoryTheory.Subobject.underlying.obj Q)
    this : P.Factors (CategoryTheory.CategoryStruct.comp f Q.arrow)
    ⊢ ExistsUnique fun h' => Eq (CategoryTheory.CategoryStruct.comp h' (P.ofLE Q h …
  -/
  refine ⟨factorThru _ _ this, ?_, fun g (hg : g ≫ _ = f) => ?_⟩
    /-
      case refine_1
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      𝒢 : Set C
      h𝒢 : CategoryTheory.IsDetecting 𝒢
      X : C
      P Q : CategoryTheory.Subobject X
      h₁ : LE.le P Q
      h₂ : ∀ (G : C), Membership.mem 𝒢 G → ∀ {f : Quiver.Hom G X}, Q.Factors f → P.F …
      G : C
      hG : Membership.mem 𝒢 G
      f : Quiver.Hom G (CategoryTheory.Subobject.underlying.obj Q)
      this : P.Factors (CategoryTheory.CategoryStruct.comp f Q.arrow)
      ⊢ (fun h' => Eq (CategoryTheory.CategoryStruct.comp h' (P.ofLE Q h₁)) f) (P.fa …
    -/
  · simp only [← cancel_mono Q.arrow, Category.assoc, ofLE_arrow, factorThru_arrow]
    /-
      🎉 no goals
    -/
  · simp only [← cancel_mono (Subobject.ofLE _ _ h₁), ← cancel_mono Q.arrow, hg, Category.assoc,
      ofLE_arrow, factorThru_arrow]


theorem inf_eq_of_isDetecting [HasPullbacks C] {𝒢 : Set C} (h𝒢 : IsDetecting 𝒢) {X : C}
    (P Q : Subobject X) (h : ∀ G ∈ 𝒢, ∀ {f : G ⟶ X}, P.Factors f → Q.Factors f) : P ⊓ Q = P :=
  eq_of_le_of_isDetecting h𝒢 _ _ _root_.inf_le_left
    fun _ hG _ hf => (inf_factors _).2 ⟨hf, h _ hG hf⟩


theorem eq_of_isDetecting [HasPullbacks C] {𝒢 : Set C} (h𝒢 : IsDetecting 𝒢) {X : C}
    (P Q : Subobject X) (h : ∀ G ∈ 𝒢, ∀ {f : G ⟶ X}, P.Factors f ↔ Q.Factors f) : P = Q :=
  calc
    P = P ⊓ Q := Eq.symm <| inf_eq_of_isDetecting h𝒢 _ _ fun G hG _ hf => (h G hG).1 hf
    _ = Q ⊓ P := inf_comm ..
    _ = Q := inf_eq_of_isDetecting h𝒢 _ _ fun G hG _ hf => (h G hG).2 hf


/-- A category with pullbacks and a small detecting set is well-powered. -/
theorem wellPowered_of_isDetecting [HasPullbacks C] {𝒢 : Set C} [Small.{w} 𝒢]
    [LocallySmall.{w} C] (h𝒢 : IsDetecting 𝒢) : WellPowered.{w} C :=
  ⟨fun X =>
    @small_of_injective _ _ _ (fun P : Subobject X => { f : ΣG : 𝒢, G.1 ⟶ X | P.Factors f.2 })
      fun P Q h => Subobject.eq_of_isDetecting h𝒢 _ _
            /-
              C : Type u₁
              inst✝³ : CategoryTheory.Category.{v₁, u₁} C
              inst✝² : CategoryTheory.Limits.HasPullbacks C
              𝒢 : Set C
              inst✝¹ : Small.{w, u₁} ↑𝒢
              inst✝ : CategoryTheory.LocallySmall.{w, v₁, u₁} C
              h𝒢 : CategoryTheory.IsDetecting 𝒢
              X : C
              P Q : CategoryTheory.Subobject X
              h : Eq ((fun P => setOf fun f => P.Factors f.snd) P) ((fun P => setOf fun f => …
              ⊢ ∀ (G : C), Membership.mem 𝒢 G → ∀ {f : Quiver.Hom G X}, Iff (P.Factors f) (Q …
            -/
        (by simpa [Set.ext_iff, Sigma.forall] using h)⟩
            /-
              🎉 no goals
            -/


theorem isCoseparating_proj_preimage {𝒢 : Set C} (h𝒢 : IsCoseparating 𝒢) :
    IsCoseparating ((proj S T).obj ⁻¹' 𝒢) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    S : D
    T : CategoryTheory.Functor C D
    𝒢 : Set C
    h𝒢 : CategoryTheory.IsCoseparating 𝒢
    ⊢ CategoryTheory.IsCoseparating (Set.preimage (CategoryTheory.StructuredArrow. …
  -/
  refine fun X Y f g hfg => ext _ _ (h𝒢 _ _ fun G hG h => ?_)
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    S : D
    T : CategoryTheory.Functor C D
    𝒢 : Set C
    h𝒢 : CategoryTheory.IsCoseparating 𝒢
    X Y : CategoryTheory.StructuredArrow S T
    f g : Quiver.Hom X Y
    hfg : ∀ (G : CategoryTheory.StructuredArrow S T), Membership.mem (Set.preimage …
    G : C
    hG : Membership.mem 𝒢 G
    h : Quiver.Hom Y.right G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f.right h) (CategoryTheory.CategorySt …
  -/
  exact congr_arg CommaMorphism.right (hfg (mk (Y.hom ≫ T.map h)) hG (homMk h rfl))
  /-
    🎉 no goals
  -/


theorem isSeparating_proj_preimage {𝒢 : Set C} (h𝒢 : IsSeparating 𝒢) :
    IsSeparating ((proj S T).obj ⁻¹' 𝒢) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    S : CategoryTheory.Functor C D
    T : D
    𝒢 : Set C
    h𝒢 : CategoryTheory.IsSeparating 𝒢
    ⊢ CategoryTheory.IsSeparating (Set.preimage (CategoryTheory.CostructuredArrow. …
  -/
  refine fun X Y f g hfg => ext _ _ (h𝒢 _ _ fun G hG h => ?_)
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    S : CategoryTheory.Functor C D
    T : D
    𝒢 : Set C
    h𝒢 : CategoryTheory.IsSeparating 𝒢
    X Y : CategoryTheory.CostructuredArrow S T
    f g : Quiver.Hom X Y
    hfg : ∀ (G : CategoryTheory.CostructuredArrow S T), Membership.mem (Set.preima …
    G : C
    hG : Membership.mem 𝒢 G
    h : Quiver.Hom G X.left
    ⊢ Eq (CategoryTheory.CategoryStruct.comp h f.left) (CategoryTheory.CategoryStr …
  -/
  exact congr_arg CommaMorphism.left (hfg (mk (S.map h ≫ X.hom)) hG (homMk h rfl))
  /-
    🎉 no goals
  -/


/-- We say that `G` is a separator if the functor `C(G, -)` is faithful. -/
def IsSeparator (G : C) : Prop :=
  IsSeparating ({G} : Set C)


/-- We say that `G` is a coseparator if the functor `C(-, G)` is faithful. -/
def IsCoseparator (G : C) : Prop :=
  IsCoseparating ({G} : Set C)


/-- We say that `G` is a detector if the functor `C(G, -)` reflects isomorphisms. -/
def IsDetector (G : C) : Prop :=
  IsDetecting ({G} : Set C)


/-- We say that `G` is a codetector if the functor `C(-, G)` reflects isomorphisms. -/
def IsCodetector (G : C) : Prop :=
  IsCodetecting ({G} : Set C)



theorem IsSeparator.of_equivalence {G : C} (h : IsSeparator G) (α : C ≌ D) :
                                        /-
                                          C : Type u₁
                                          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                          D : Type u₂
                                          inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                          G : C
                                          h : CategoryTheory.IsSeparator G
                                          α : CategoryTheory.Equivalence C D
                                          ⊢ CategoryTheory.IsSeparator (α.functor.obj G)
                                        -/
    IsSeparator (α.functor.obj G) := by simpa using IsSeparating.of_equivalence h α
                                        /-
                                          🎉 no goals
                                        -/


theorem IsCoseparator.of_equivalence {G : C} (h : IsCoseparator G) (α : C ≌ D) :
                                          /-
                                            C : Type u₁
                                            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                            D : Type u₂
                                            inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                            G : C
                                            h : CategoryTheory.IsCoseparator G
                                            α : CategoryTheory.Equivalence C D
                                            ⊢ CategoryTheory.IsCoseparator (α.functor.obj G)
                                          -/
    IsCoseparator (α.functor.obj G) := by simpa using IsCoseparating.of_equivalence h α
                                          /-
                                            🎉 no goals
                                          -/


theorem isSeparator_op_iff (G : C) : IsSeparator (op G) ↔ IsCoseparator G := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    G : C
    ⊢ Iff (CategoryTheory.IsSeparator { unop := G }) (CategoryTheory.IsCoseparator …
  -/
  rw [IsSeparator, IsCoseparator, ← isSeparating_op_iff, Set.singleton_op]
  /-
    🎉 no goals
  -/


theorem isCoseparator_op_iff (G : C) : IsCoseparator (op G) ↔ IsSeparator G := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    G : C
    ⊢ Iff (CategoryTheory.IsCoseparator { unop := G }) (CategoryTheory.IsSeparator …
  -/
  rw [IsSeparator, IsCoseparator, ← isCoseparating_op_iff, Set.singleton_op]
  /-
    🎉 no goals
  -/


theorem isCoseparator_unop_iff (G : Cᵒᵖ) : IsCoseparator (unop G) ↔ IsSeparator G := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    G : Opposite C
    ⊢ Iff (CategoryTheory.IsCoseparator (Opposite.unop G)) (CategoryTheory.IsSepar …
  -/
  rw [IsSeparator, IsCoseparator, ← isCoseparating_unop_iff, Set.singleton_unop]
  /-
    🎉 no goals
  -/


theorem isSeparator_unop_iff (G : Cᵒᵖ) : IsSeparator (unop G) ↔ IsCoseparator G := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    G : Opposite C
    ⊢ Iff (CategoryTheory.IsSeparator (Opposite.unop G)) (CategoryTheory.IsCosepar …
  -/
  rw [IsSeparator, IsCoseparator, ← isSeparating_unop_iff, Set.singleton_unop]
  /-
    🎉 no goals
  -/


theorem isDetector_op_iff (G : C) : IsDetector (op G) ↔ IsCodetector G := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    G : C
    ⊢ Iff (CategoryTheory.IsDetector { unop := G }) (CategoryTheory.IsCodetector G)
  -/
  rw [IsDetector, IsCodetector, ← isDetecting_op_iff, Set.singleton_op]
  /-
    🎉 no goals
  -/


theorem isCodetector_op_iff (G : C) : IsCodetector (op G) ↔ IsDetector G := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    G : C
    ⊢ Iff (CategoryTheory.IsCodetector { unop := G }) (CategoryTheory.IsDetector G)
  -/
  rw [IsDetector, IsCodetector, ← isCodetecting_op_iff, Set.singleton_op]
  /-
    🎉 no goals
  -/


theorem isCodetector_unop_iff (G : Cᵒᵖ) : IsCodetector (unop G) ↔ IsDetector G := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    G : Opposite C
    ⊢ Iff (CategoryTheory.IsCodetector (Opposite.unop G)) (CategoryTheory.IsDetect …
  -/
  rw [IsDetector, IsCodetector, ← isCodetecting_unop_iff, Set.singleton_unop]
  /-
    🎉 no goals
  -/


theorem isDetector_unop_iff (G : Cᵒᵖ) : IsDetector (unop G) ↔ IsCodetector G := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    G : Opposite C
    ⊢ Iff (CategoryTheory.IsDetector (Opposite.unop G)) (CategoryTheory.IsCodetect …
  -/
  rw [IsDetector, IsCodetector, ← isDetecting_unop_iff, Set.singleton_unop]
  /-
    🎉 no goals
  -/


theorem IsDetector.isSeparator [HasEqualizers C] {G : C} : IsDetector G → IsSeparator G :=
  IsDetecting.isSeparating


theorem IsCodetector.isCoseparator [HasCoequalizers C] {G : C} : IsCodetector G → IsCoseparator G :=
  IsCodetecting.isCoseparating


theorem IsSeparator.isDetector [Balanced C] {G : C} : IsSeparator G → IsDetector G :=
  IsSeparating.isDetecting


theorem IsCoseparator.isCodetector [Balanced C] {G : C} : IsCoseparator G → IsCodetector G :=
  IsCoseparating.isCodetecting


theorem isSeparator_def (G : C) :
    IsSeparator G ↔ ∀ ⦃X Y : C⦄ (f g : X ⟶ Y), (∀ h : G ⟶ X, h ≫ f = h ≫ g) → f = g :=
  ⟨fun hG X Y f g hfg =>
    hG _ _ fun H hH h => by
      /-
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        G : C
        hG : CategoryTheory.IsSeparator G
        X Y : C
        f g : Quiver.Hom X Y
        hfg : ∀ (h : Quiver.Hom G X), Eq (CategoryTheory.CategoryStruct.comp h f) (Cat …
        H : C
        hH : Membership.mem (Singleton.singleton G) H
        h : Quiver.Hom H X
        ⊢ Eq (CategoryTheory.CategoryStruct.comp h f) (CategoryTheory.CategoryStruct.c …
      -/
      obtain rfl := Set.mem_singleton_iff.1 hH
      /-
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        X Y : C
        f g : Quiver.Hom X Y
        H : C
        h : Quiver.Hom H X
        hG : CategoryTheory.IsSeparator H
        hfg : ∀ (h : Quiver.Hom H X), Eq (CategoryTheory.CategoryStruct.comp h f) (Cat …
        hH : Membership.mem (Singleton.singleton H) H
        ⊢ Eq (CategoryTheory.CategoryStruct.comp h f) (CategoryTheory.CategoryStruct.c …
      -/
      exact hfg h,
      /-
        🎉 no goals
      -/
    fun hG _ _ _ _ hfg => hG _ _ fun _ => hfg _ (Set.mem_singleton _) _⟩


theorem IsSeparator.def {G : C} :
    IsSeparator G → ∀ ⦃X Y : C⦄ (f g : X ⟶ Y), (∀ h : G ⟶ X, h ≫ f = h ≫ g) → f = g :=
  (isSeparator_def _).1


theorem isCoseparator_def (G : C) :
    IsCoseparator G ↔ ∀ ⦃X Y : C⦄ (f g : X ⟶ Y), (∀ h : Y ⟶ G, f ≫ h = g ≫ h) → f = g :=
  ⟨fun hG X Y f g hfg =>
    hG _ _ fun H hH h => by
      /-
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        G : C
        hG : CategoryTheory.IsCoseparator G
        X Y : C
        f g : Quiver.Hom X Y
        hfg : ∀ (h : Quiver.Hom Y G), Eq (CategoryTheory.CategoryStruct.comp f h) (Cat …
        H : C
        hH : Membership.mem (Singleton.singleton G) H
        h : Quiver.Hom Y H
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStruct.c …
      -/
      obtain rfl := Set.mem_singleton_iff.1 hH
      /-
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        X Y : C
        f g : Quiver.Hom X Y
        H : C
        h : Quiver.Hom Y H
        hG : CategoryTheory.IsCoseparator H
        hfg : ∀ (h : Quiver.Hom Y H), Eq (CategoryTheory.CategoryStruct.comp f h) (Cat …
        hH : Membership.mem (Singleton.singleton H) H
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStruct.c …
      -/
      exact hfg h,
      /-
        🎉 no goals
      -/
    fun hG _ _ _ _ hfg => hG _ _ fun _ => hfg _ (Set.mem_singleton _) _⟩


theorem IsCoseparator.def {G : C} :
    IsCoseparator G → ∀ ⦃X Y : C⦄ (f g : X ⟶ Y), (∀ h : Y ⟶ G, f ≫ h = g ≫ h) → f = g :=
  (isCoseparator_def _).1


theorem isDetector_def (G : C) :
    IsDetector G ↔ ∀ ⦃X Y : C⦄ (f : X ⟶ Y), (∀ h : G ⟶ Y, ∃! h', h' ≫ f = h) → IsIso f :=
  ⟨fun hG X Y f hf =>
    hG _ fun H hH h => by
      /-
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        G : C
        hG : CategoryTheory.IsDetector G
        X Y : C
        f : Quiver.Hom X Y
        hf : ∀ (h : Quiver.Hom G Y), ExistsUnique fun h' => Eq (CategoryTheory.Categor …
        H : C
        hH : Membership.mem (Singleton.singleton G) H
        h : Quiver.Hom H Y
        ⊢ ExistsUnique fun h' => Eq (CategoryTheory.CategoryStruct.comp h' f) h
      -/
      obtain rfl := Set.mem_singleton_iff.1 hH
      /-
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        X Y : C
        f : Quiver.Hom X Y
        H : C
        h : Quiver.Hom H Y
        hG : CategoryTheory.IsDetector H
        hf : ∀ (h : Quiver.Hom H Y), ExistsUnique fun h' => Eq (CategoryTheory.Categor …
        hH : Membership.mem (Singleton.singleton H) H
        ⊢ ExistsUnique fun h' => Eq (CategoryTheory.CategoryStruct.comp h' f) h
      -/
      exact hf h,
      /-
        🎉 no goals
      -/
    fun hG _ _ _ hf => hG _ fun _ => hf _ (Set.mem_singleton _) _⟩


theorem IsDetector.def {G : C} :
    IsDetector G → ∀ ⦃X Y : C⦄ (f : X ⟶ Y), (∀ h : G ⟶ Y, ∃! h', h' ≫ f = h) → IsIso f :=
  (isDetector_def _).1


theorem isCodetector_def (G : C) :
    IsCodetector G ↔ ∀ ⦃X Y : C⦄ (f : X ⟶ Y), (∀ h : X ⟶ G, ∃! h', f ≫ h' = h) → IsIso f :=
  ⟨fun hG X Y f hf =>
    hG _ fun H hH h => by
      /-
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        G : C
        hG : CategoryTheory.IsCodetector G
        X Y : C
        f : Quiver.Hom X Y
        hf : ∀ (h : Quiver.Hom X G), ExistsUnique fun h' => Eq (CategoryTheory.Categor …
        H : C
        hH : Membership.mem (Singleton.singleton G) H
        h : Quiver.Hom X H
        ⊢ ExistsUnique fun h' => Eq (CategoryTheory.CategoryStruct.comp f h') h
      -/
      obtain rfl := Set.mem_singleton_iff.1 hH
      /-
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        X Y : C
        f : Quiver.Hom X Y
        H : C
        h : Quiver.Hom X H
        hG : CategoryTheory.IsCodetector H
        hf : ∀ (h : Quiver.Hom X H), ExistsUnique fun h' => Eq (CategoryTheory.Categor …
        hH : Membership.mem (Singleton.singleton H) H
        ⊢ ExistsUnique fun h' => Eq (CategoryTheory.CategoryStruct.comp f h') h
      -/
      exact hf h,
      /-
        🎉 no goals
      -/
    fun hG _ _ _ hf => hG _ fun _ => hf _ (Set.mem_singleton _) _⟩


theorem IsCodetector.def {G : C} :
    IsCodetector G → ∀ ⦃X Y : C⦄ (f : X ⟶ Y), (∀ h : X ⟶ G, ∃! h', f ≫ h' = h) → IsIso f :=
  (isCodetector_def _).1


theorem isSeparator_iff_faithful_coyoneda_obj (G : C) :
    IsSeparator G ↔ (coyoneda.obj (op G)).Faithful :=
  ⟨fun hG => ⟨fun hfg => hG.def _ _ (congr_fun hfg)⟩, fun _ =>
    (isSeparator_def _).2 fun _ _ _ _ hfg => (coyoneda.obj (op G)).map_injective (funext hfg)⟩


theorem isCoseparator_iff_faithful_yoneda_obj (G : C) : IsCoseparator G ↔ (yoneda.obj G).Faithful :=
  ⟨fun hG => ⟨fun hfg => Quiver.Hom.unop_inj (hG.def _ _ (congr_fun hfg))⟩, fun _ =>
    (isCoseparator_def _).2 fun _ _ _ _ hfg =>
      Quiver.Hom.op_inj <| (yoneda.obj G).map_injective (funext hfg)⟩


theorem isSeparator_iff_epi (G : C) [∀ A : C, HasCoproduct fun _ : G ⟶ A => G] :
    IsSeparator G ↔ ∀ A : C, Epi (Sigma.desc fun f : G ⟶ A => f) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    G : C
    inst✝ : ∀ (A : C), CategoryTheory.Limits.HasCoproduct fun x => G
    ⊢ Iff (CategoryTheory.IsSeparator G) (∀ (A : C), CategoryTheory.Epi (CategoryT …
  -/
  rw [isSeparator_def]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    G : C
    inst✝ : ∀ (A : C), CategoryTheory.Limits.HasCoproduct fun x => G
    ⊢ Iff (∀ ⦃X Y : C⦄ (f g : Quiver.Hom X Y), (∀ (h : Quiver.Hom G X), Eq (Catego …
  -/
  refine ⟨fun h A => ⟨fun u v huv => h _ _ fun i => ?_⟩, fun h X Y f g hh => ?_⟩
    /-
      case refine_1
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      G : C
      inst✝ : ∀ (A : C), CategoryTheory.Limits.HasCoproduct fun x => G
      h : ∀ ⦃X Y : C⦄ (f g : Quiver.Hom X Y), (∀ (h : Quiver.Hom G X), Eq (CategoryT …
      A Z✝ : C
      u v : Quiver.Hom A Z✝
      huv : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.desc …
      i : Quiver.Hom G A
      ⊢ Eq (CategoryTheory.CategoryStruct.comp i u) (CategoryTheory.CategoryStruct.c …
    -/
  · simpa using Sigma.ι _ i ≫= huv
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      G : C
      inst✝ : ∀ (A : C), CategoryTheory.Limits.HasCoproduct fun x => G
      h : ∀ (A : C), CategoryTheory.Epi (CategoryTheory.Limits.Sigma.desc fun f => f)
      X Y : C
      f g : Quiver.Hom X Y
      hh : ∀ (h : Quiver.Hom G X), Eq (CategoryTheory.CategoryStruct.comp h f) (Cate …
      ⊢ Eq f g
    -/
  · haveI := h X
    /-
      case refine_2
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      G : C
      inst✝ : ∀ (A : C), CategoryTheory.Limits.HasCoproduct fun x => G
      h : ∀ (A : C), CategoryTheory.Epi (CategoryTheory.Limits.Sigma.desc fun f => f)
      X Y : C
      f g : Quiver.Hom X Y
      hh : ∀ (h : Quiver.Hom G X), Eq (CategoryTheory.CategoryStruct.comp h f) (Cate …
      this : CategoryTheory.Epi (CategoryTheory.Limits.Sigma.desc fun f => f)
      ⊢ Eq f g
    -/
    refine (cancel_epi (Sigma.desc fun f : G ⟶ X => f)).1 (colimit.hom_ext fun j => ?_)
    /-
      case refine_2
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      G : C
      inst✝ : ∀ (A : C), CategoryTheory.Limits.HasCoproduct fun x => G
      h : ∀ (A : C), CategoryTheory.Epi (CategoryTheory.Limits.Sigma.desc fun f => f)
      X Y : C
      f g : Quiver.Hom X Y
      hh : ∀ (h : Quiver.Hom G X), Eq (CategoryTheory.CategoryStruct.comp h f) (Cate …
      this : CategoryTheory.Epi (CategoryTheory.Limits.Sigma.desc fun f => f)
      j : CategoryTheory.Discrete (Quiver.Hom G X)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (Cat …
    -/
    simpa using hh j.as
    /-
      🎉 no goals
    -/


theorem isCoseparator_iff_mono (G : C) [∀ A : C, HasProduct fun _ : A ⟶ G => G] :
    IsCoseparator G ↔ ∀ A : C, Mono (Pi.lift fun f : A ⟶ G => f) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    G : C
    inst✝ : ∀ (A : C), CategoryTheory.Limits.HasProduct fun x => G
    ⊢ Iff (CategoryTheory.IsCoseparator G) (∀ (A : C), CategoryTheory.Mono (Catego …
  -/
  rw [isCoseparator_def]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    G : C
    inst✝ : ∀ (A : C), CategoryTheory.Limits.HasProduct fun x => G
    ⊢ Iff (∀ ⦃X Y : C⦄ (f g : Quiver.Hom X Y), (∀ (h : Quiver.Hom Y G), Eq (Catego …
  -/
  refine ⟨fun h A => ⟨fun u v huv => h _ _ fun i => ?_⟩, fun h X Y f g hh => ?_⟩
    /-
      case refine_1
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      G : C
      inst✝ : ∀ (A : C), CategoryTheory.Limits.HasProduct fun x => G
      h : ∀ ⦃X Y : C⦄ (f g : Quiver.Hom X Y), (∀ (h : Quiver.Hom Y G), Eq (CategoryT …
      A Z✝ : C
      u v : Quiver.Hom Z✝ A
      huv : Eq (CategoryTheory.CategoryStruct.comp u (CategoryTheory.Limits.Pi.lift  …
      i : Quiver.Hom A G
      ⊢ Eq (CategoryTheory.CategoryStruct.comp u i) (CategoryTheory.CategoryStruct.c …
    -/
  · simpa using huv =≫ Pi.π _ i
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      G : C
      inst✝ : ∀ (A : C), CategoryTheory.Limits.HasProduct fun x => G
      h : ∀ (A : C), CategoryTheory.Mono (CategoryTheory.Limits.Pi.lift fun f => f)
      X Y : C
      f g : Quiver.Hom X Y
      hh : ∀ (h : Quiver.Hom Y G), Eq (CategoryTheory.CategoryStruct.comp f h) (Cate …
      ⊢ Eq f g
    -/
  · haveI := h Y
    /-
      case refine_2
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      G : C
      inst✝ : ∀ (A : C), CategoryTheory.Limits.HasProduct fun x => G
      h : ∀ (A : C), CategoryTheory.Mono (CategoryTheory.Limits.Pi.lift fun f => f)
      X Y : C
      f g : Quiver.Hom X Y
      hh : ∀ (h : Quiver.Hom Y G), Eq (CategoryTheory.CategoryStruct.comp f h) (Cate …
      this : CategoryTheory.Mono (CategoryTheory.Limits.Pi.lift fun f => f)
      ⊢ Eq f g
    -/
    refine (cancel_mono (Pi.lift fun f : Y ⟶ G => f)).1 (limit.hom_ext fun j => ?_)
    /-
      case refine_2
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      G : C
      inst✝ : ∀ (A : C), CategoryTheory.Limits.HasProduct fun x => G
      h : ∀ (A : C), CategoryTheory.Mono (CategoryTheory.Limits.Pi.lift fun f => f)
      X Y : C
      f g : Quiver.Hom X Y
      hh : ∀ (h : Quiver.Hom Y G), Eq (CategoryTheory.CategoryStruct.comp f h) (Cate …
      this : CategoryTheory.Mono (CategoryTheory.Limits.Pi.lift fun f => f)
      j : CategoryTheory.Discrete (Quiver.Hom Y G)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
    -/
    simpa using hh j.as
    /-
      🎉 no goals
    -/


theorem isSeparator_coprod (G H : C) [HasBinaryCoproduct G H] :
    IsSeparator (G ⨿ H) ↔ IsSeparating ({G, H} : Set C) := by
  refine
    ⟨fun h X Y u v huv => ?_, fun h =>
      (isSeparator_def _).2 fun X Y u v huv => h _ _ fun Z hZ g => ?_⟩
    /-
      case refine_1
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      G H : C
      inst✝ : CategoryTheory.Limits.HasBinaryCoproduct G H
      h : CategoryTheory.IsSeparator (CategoryTheory.Limits.coprod G H)
      X Y : C
      u v : Quiver.Hom X Y
      huv : ∀ (G_1 : C), Membership.mem (Insert.insert G (Singleton.singleton H)) G_ …
      ⊢ Eq u v
    -/
  · refine h.def _ _ fun g => coprod.hom_ext ?_ ?_
      /-
        case refine_1.refine_1
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
        G H : C
        inst✝ : CategoryTheory.Limits.HasBinaryCoproduct G H
        h : CategoryTheory.IsSeparator (CategoryTheory.Limits.coprod G H)
        X Y : C
        u v : Quiver.Hom X Y
        huv : ∀ (G_1 : C), Membership.mem (Insert.insert G (Singleton.singleton H)) G_ …
        g : Quiver.Hom (CategoryTheory.Limits.coprod G H) X
        ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.coprod.inl (Cat …
      -/
    · simpa using huv G (by simp) (coprod.inl ≫ g)
      /-
        🎉 no goals
      -/
      /-
        case refine_1.refine_2
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
        G H : C
        inst✝ : CategoryTheory.Limits.HasBinaryCoproduct G H
        h : CategoryTheory.IsSeparator (CategoryTheory.Limits.coprod G H)
        X Y : C
        u v : Quiver.Hom X Y
        huv : ∀ (G_1 : C), Membership.mem (Insert.insert G (Singleton.singleton H)) G_ …
        g : Quiver.Hom (CategoryTheory.Limits.coprod G H) X
        ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.coprod.inr (Cat …
      -/
    · simpa using huv H (by simp) (coprod.inr ≫ g)
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      G H : C
      inst✝ : CategoryTheory.Limits.HasBinaryCoproduct G H
      h : CategoryTheory.IsSeparating (Insert.insert G (Singleton.singleton H))
      X Y : C
      u v : Quiver.Hom X Y
      huv : ∀ (h : Quiver.Hom (CategoryTheory.Limits.coprod G H) X), Eq (CategoryThe …
      Z : C
      hZ : Membership.mem (Insert.insert G (Singleton.singleton H)) Z
      g : Quiver.Hom Z X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp g u) (CategoryTheory.CategoryStruct.c …
    -/
  · simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at hZ
    /-
      case refine_2
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      G H : C
      inst✝ : CategoryTheory.Limits.HasBinaryCoproduct G H
      h : CategoryTheory.IsSeparating (Insert.insert G (Singleton.singleton H))
      X Y : C
      u v : Quiver.Hom X Y
      huv : ∀ (h : Quiver.Hom (CategoryTheory.Limits.coprod G H) X), Eq (CategoryThe …
      Z : C
      g : Quiver.Hom Z X
      hZ : Or (Eq Z G) (Eq Z H)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp g u) (CategoryTheory.CategoryStruct.c …
    -/
    rcases hZ with (rfl | rfl)
      /-
        case refine_2.inl
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
        H X Y : C
        u v : Quiver.Hom X Y
        Z : C
        g : Quiver.Hom Z X
        inst✝ : CategoryTheory.Limits.HasBinaryCoproduct Z H
        h : CategoryTheory.IsSeparating (Insert.insert Z (Singleton.singleton H))
        huv : ∀ (h : Quiver.Hom (CategoryTheory.Limits.coprod Z H) X), Eq (CategoryThe …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp g u) (CategoryTheory.CategoryStruct.c …
      -/
    · simpa using coprod.inl ≫= huv (coprod.desc g 0)
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
        G X Y : C
        u v : Quiver.Hom X Y
        Z : C
        g : Quiver.Hom Z X
        inst✝ : CategoryTheory.Limits.HasBinaryCoproduct G Z
        h : CategoryTheory.IsSeparating (Insert.insert G (Singleton.singleton Z))
        huv : ∀ (h : Quiver.Hom (CategoryTheory.Limits.coprod G Z) X), Eq (CategoryThe …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp g u) (CategoryTheory.CategoryStruct.c …
      -/
    · simpa using coprod.inr ≫= huv (coprod.desc 0 g)
      /-
        🎉 no goals
      -/


theorem isSeparator_coprod_of_isSeparator_left (G H : C) [HasBinaryCoproduct G H]
    (hG : IsSeparator G) : IsSeparator (G ⨿ H) :=
                                                           /-
                                                             C : Type u₁
                                                             inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                             inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                             G H : C
                                                             inst✝ : CategoryTheory.Limits.HasBinaryCoproduct G H
                                                             hG : CategoryTheory.IsSeparator G
                                                             ⊢ HasSubset.Subset (Singleton.singleton G) (Insert.insert G (Singleton.singlet …
                                                           -/
  (isSeparator_coprod _ _).2 <| IsSeparating.mono hG <| by simp
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem isSeparator_coprod_of_isSeparator_right (G H : C) [HasBinaryCoproduct G H]
    (hH : IsSeparator H) : IsSeparator (G ⨿ H) :=
                                                           /-
                                                             C : Type u₁
                                                             inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                             inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                             G H : C
                                                             inst✝ : CategoryTheory.Limits.HasBinaryCoproduct G H
                                                             hH : CategoryTheory.IsSeparator H
                                                             ⊢ HasSubset.Subset (Singleton.singleton H) (Insert.insert G (Singleton.singlet …
                                                           -/
  (isSeparator_coprod _ _).2 <| IsSeparating.mono hH <| by simp
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem isSeparator_sigma {β : Type w} (f : β → C) [HasCoproduct f] :
    IsSeparator (∐ f) ↔ IsSeparating (Set.range f) := by
  refine
    ⟨fun h X Y u v huv => ?_, fun h =>
      (isSeparator_def _).2 fun X Y u v huv => h _ _ fun Z hZ g => ?_⟩
    /-
      case refine_1
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      β : Type w
      f : β → C
      inst✝ : CategoryTheory.Limits.HasCoproduct f
      h : CategoryTheory.IsSeparator (CategoryTheory.Limits.sigmaObj f)
      X Y : C
      u v : Quiver.Hom X Y
      huv : ∀ (G : C), Membership.mem (Set.range f) G → ∀ (h : Quiver.Hom G X), Eq ( …
      ⊢ Eq u v
    -/
  · refine h.def _ _ fun g => colimit.hom_ext fun b => ?_
    /-
      case refine_1
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      β : Type w
      f : β → C
      inst✝ : CategoryTheory.Limits.HasCoproduct f
      h : CategoryTheory.IsSeparator (CategoryTheory.Limits.sigmaObj f)
      X Y : C
      u v : Quiver.Hom X Y
      huv : ∀ (G : C), Membership.mem (Set.range f) G → ∀ (h : Quiver.Hom G X), Eq ( …
      g : Quiver.Hom (CategoryTheory.Limits.sigmaObj f) X
      b : CategoryTheory.Discrete β
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (Cat …
    -/
    simpa using huv (f b.as) (by simp) (colimit.ι (Discrete.functor f) _ ≫ g)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      β : Type w
      f : β → C
      inst✝ : CategoryTheory.Limits.HasCoproduct f
      h : CategoryTheory.IsSeparating (Set.range f)
      X Y : C
      u v : Quiver.Hom X Y
      huv : ∀ (h : Quiver.Hom (CategoryTheory.Limits.sigmaObj f) X), Eq (CategoryThe …
      Z : C
      hZ : Membership.mem (Set.range f) Z
      g : Quiver.Hom Z X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp g u) (CategoryTheory.CategoryStruct.c …
    -/
  · obtain ⟨b, rfl⟩ := Set.mem_range.1 hZ
    /-
      case refine_2.intro
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      β : Type w
      f : β → C
      inst✝ : CategoryTheory.Limits.HasCoproduct f
      h : CategoryTheory.IsSeparating (Set.range f)
      X Y : C
      u v : Quiver.Hom X Y
      huv : ∀ (h : Quiver.Hom (CategoryTheory.Limits.sigmaObj f) X), Eq (CategoryThe …
      b : β
      hZ : Membership.mem (Set.range f) (f b)
      g : Quiver.Hom (f b) X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp g u) (CategoryTheory.CategoryStruct.c …
    -/
    classical simpa using Sigma.ι f b ≫= huv (Sigma.desc (Pi.single b g))
    /-
      🎉 no goals
    -/


theorem isSeparator_sigma_of_isSeparator {β : Type w} (f : β → C) [HasCoproduct f] (b : β)
    (hb : IsSeparator (f b)) : IsSeparator (∐ f) :=
                                                        /-
                                                          C : Type u₁
                                                          inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                          β : Type w
                                                          f : β → C
                                                          inst✝ : CategoryTheory.Limits.HasCoproduct f
                                                          b : β
                                                          hb : CategoryTheory.IsSeparator (f b)
                                                          ⊢ HasSubset.Subset (Singleton.singleton (f b)) (Set.range f)
                                                        -/
  (isSeparator_sigma _).2 <| IsSeparating.mono hb <| by simp
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem isCoseparator_prod (G H : C) [HasBinaryProduct G H] :
    IsCoseparator (G ⨯ H) ↔ IsCoseparating ({G, H} : Set C) := by
  refine
    ⟨fun h X Y u v huv => ?_, fun h =>
      (isCoseparator_def _).2 fun X Y u v huv => h _ _ fun Z hZ g => ?_⟩
    /-
      case refine_1
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      G H : C
      inst✝ : CategoryTheory.Limits.HasBinaryProduct G H
      h : CategoryTheory.IsCoseparator (CategoryTheory.Limits.prod G H)
      X Y : C
      u v : Quiver.Hom X Y
      huv : ∀ (G_1 : C), Membership.mem (Insert.insert G (Singleton.singleton H)) G_ …
      ⊢ Eq u v
    -/
  · refine h.def _ _ fun g => Limits.prod.hom_ext ?_ ?_
      /-
        case refine_1.refine_1
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
        G H : C
        inst✝ : CategoryTheory.Limits.HasBinaryProduct G H
        h : CategoryTheory.IsCoseparator (CategoryTheory.Limits.prod G H)
        X Y : C
        u v : Quiver.Hom X Y
        huv : ∀ (G_1 : C), Membership.mem (Insert.insert G (Singleton.singleton H)) G_ …
        g : Quiver.Hom Y (CategoryTheory.Limits.prod G H)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp u …
      -/
    · simpa using huv G (by simp) (g ≫ Limits.prod.fst)
      /-
        🎉 no goals
      -/
      /-
        case refine_1.refine_2
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
        G H : C
        inst✝ : CategoryTheory.Limits.HasBinaryProduct G H
        h : CategoryTheory.IsCoseparator (CategoryTheory.Limits.prod G H)
        X Y : C
        u v : Quiver.Hom X Y
        huv : ∀ (G_1 : C), Membership.mem (Insert.insert G (Singleton.singleton H)) G_ …
        g : Quiver.Hom Y (CategoryTheory.Limits.prod G H)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp u …
      -/
    · simpa using huv H (by simp) (g ≫ Limits.prod.snd)
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      G H : C
      inst✝ : CategoryTheory.Limits.HasBinaryProduct G H
      h : CategoryTheory.IsCoseparating (Insert.insert G (Singleton.singleton H))
      X Y : C
      u v : Quiver.Hom X Y
      huv : ∀ (h : Quiver.Hom Y (CategoryTheory.Limits.prod G H)), Eq (CategoryTheor …
      Z : C
      hZ : Membership.mem (Insert.insert G (Singleton.singleton H)) Z
      g : Quiver.Hom Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp u g) (CategoryTheory.CategoryStruct.c …
    -/
  · simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at hZ
    /-
      case refine_2
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      G H : C
      inst✝ : CategoryTheory.Limits.HasBinaryProduct G H
      h : CategoryTheory.IsCoseparating (Insert.insert G (Singleton.singleton H))
      X Y : C
      u v : Quiver.Hom X Y
      huv : ∀ (h : Quiver.Hom Y (CategoryTheory.Limits.prod G H)), Eq (CategoryTheor …
      Z : C
      g : Quiver.Hom Y Z
      hZ : Or (Eq Z G) (Eq Z H)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp u g) (CategoryTheory.CategoryStruct.c …
    -/
    rcases hZ with (rfl | rfl)
      /-
        case refine_2.inl
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
        H X Y : C
        u v : Quiver.Hom X Y
        Z : C
        g : Quiver.Hom Y Z
        inst✝ : CategoryTheory.Limits.HasBinaryProduct Z H
        h : CategoryTheory.IsCoseparating (Insert.insert Z (Singleton.singleton H))
        huv : ∀ (h : Quiver.Hom Y (CategoryTheory.Limits.prod Z H)), Eq (CategoryTheor …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp u g) (CategoryTheory.CategoryStruct.c …
      -/
    · simpa using huv (prod.lift g 0) =≫ Limits.prod.fst
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
        G X Y : C
        u v : Quiver.Hom X Y
        Z : C
        g : Quiver.Hom Y Z
        inst✝ : CategoryTheory.Limits.HasBinaryProduct G Z
        h : CategoryTheory.IsCoseparating (Insert.insert G (Singleton.singleton Z))
        huv : ∀ (h : Quiver.Hom Y (CategoryTheory.Limits.prod G Z)), Eq (CategoryTheor …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp u g) (CategoryTheory.CategoryStruct.c …
      -/
    · simpa using huv (prod.lift 0 g) =≫ Limits.prod.snd
      /-
        🎉 no goals
      -/


theorem isCoseparator_prod_of_isCoseparator_left (G H : C) [HasBinaryProduct G H]
    (hG : IsCoseparator G) : IsCoseparator (G ⨯ H) :=
                                                             /-
                                                               C : Type u₁
                                                               inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                               inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                               G H : C
                                                               inst✝ : CategoryTheory.Limits.HasBinaryProduct G H
                                                               hG : CategoryTheory.IsCoseparator G
                                                               ⊢ HasSubset.Subset (Singleton.singleton G) (Insert.insert G (Singleton.singlet …
                                                             -/
  (isCoseparator_prod _ _).2 <| IsCoseparating.mono hG <| by simp
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem isCoseparator_prod_of_isCoseparator_right (G H : C) [HasBinaryProduct G H]
    (hH : IsCoseparator H) : IsCoseparator (G ⨯ H) :=
                                                             /-
                                                               C : Type u₁
                                                               inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                               inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                               G H : C
                                                               inst✝ : CategoryTheory.Limits.HasBinaryProduct G H
                                                               hH : CategoryTheory.IsCoseparator H
                                                               ⊢ HasSubset.Subset (Singleton.singleton H) (Insert.insert G (Singleton.singlet …
                                                             -/
  (isCoseparator_prod _ _).2 <| IsCoseparating.mono hH <| by simp
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem isCoseparator_pi {β : Type w} (f : β → C) [HasProduct f] :
    IsCoseparator (∏ᶜ f) ↔ IsCoseparating (Set.range f) := by
  refine
    ⟨fun h X Y u v huv => ?_, fun h =>
      (isCoseparator_def _).2 fun X Y u v huv => h _ _ fun Z hZ g => ?_⟩
    /-
      case refine_1
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      β : Type w
      f : β → C
      inst✝ : CategoryTheory.Limits.HasProduct f
      h : CategoryTheory.IsCoseparator (CategoryTheory.Limits.piObj f)
      X Y : C
      u v : Quiver.Hom X Y
      huv : ∀ (G : C), Membership.mem (Set.range f) G → ∀ (h : Quiver.Hom Y G), Eq ( …
      ⊢ Eq u v
    -/
  · refine h.def _ _ fun g => limit.hom_ext fun b => ?_
    /-
      case refine_1
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      β : Type w
      f : β → C
      inst✝ : CategoryTheory.Limits.HasProduct f
      h : CategoryTheory.IsCoseparator (CategoryTheory.Limits.piObj f)
      X Y : C
      u v : Quiver.Hom X Y
      huv : ∀ (G : C), Membership.mem (Set.range f) G → ∀ (h : Quiver.Hom Y G), Eq ( …
      g : Quiver.Hom Y (CategoryTheory.Limits.piObj f)
      b : CategoryTheory.Discrete β
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp u …
    -/
    simpa using huv (f b.as) (by simp) (g ≫ limit.π (Discrete.functor f) _)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      β : Type w
      f : β → C
      inst✝ : CategoryTheory.Limits.HasProduct f
      h : CategoryTheory.IsCoseparating (Set.range f)
      X Y : C
      u v : Quiver.Hom X Y
      huv : ∀ (h : Quiver.Hom Y (CategoryTheory.Limits.piObj f)), Eq (CategoryTheory …
      Z : C
      hZ : Membership.mem (Set.range f) Z
      g : Quiver.Hom Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp u g) (CategoryTheory.CategoryStruct.c …
    -/
  · obtain ⟨b, rfl⟩ := Set.mem_range.1 hZ
    /-
      case refine_2.intro
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      β : Type w
      f : β → C
      inst✝ : CategoryTheory.Limits.HasProduct f
      h : CategoryTheory.IsCoseparating (Set.range f)
      X Y : C
      u v : Quiver.Hom X Y
      huv : ∀ (h : Quiver.Hom Y (CategoryTheory.Limits.piObj f)), Eq (CategoryTheory …
      b : β
      hZ : Membership.mem (Set.range f) (f b)
      g : Quiver.Hom Y (f b)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp u g) (CategoryTheory.CategoryStruct.c …
    -/
    classical simpa using huv (Pi.lift (Pi.single b g)) =≫ Pi.π f b
    /-
      🎉 no goals
    -/


theorem isCoseparator_pi_of_isCoseparator {β : Type w} (f : β → C) [HasProduct f] (b : β)
    (hb : IsCoseparator (f b)) : IsCoseparator (∏ᶜ f) :=
                                                         /-
                                                           C : Type u₁
                                                           inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                           inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                           β : Type w
                                                           f : β → C
                                                           inst✝ : CategoryTheory.Limits.HasProduct f
                                                           b : β
                                                           hb : CategoryTheory.IsCoseparator (f b)
                                                           ⊢ HasSubset.Subset (Singleton.singleton (f b)) (Set.range f)
                                                         -/
  (isCoseparator_pi _).2 <| IsCoseparating.mono hb <| by simp
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem isDetector_iff_reflectsIsomorphisms_coyoneda_obj (G : C) :
    IsDetector G ↔ (coyoneda.obj (op G)).ReflectsIsomorphisms := by
  refine
    ⟨fun hG => ⟨fun f hf => hG.def _ fun h => ?_⟩, fun h =>
      (isDetector_def _).2 fun X Y f hf => ?_⟩
    /-
      case refine_1
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      G : C
      hG : CategoryTheory.IsDetector G
      A✝ B✝ : C
      f : Quiver.Hom A✝ B✝
      hf : CategoryTheory.IsIso ((CategoryTheory.coyoneda.obj { unop := G }).map f)
      h : Quiver.Hom G B✝
      ⊢ ExistsUnique fun h' => Eq (CategoryTheory.CategoryStruct.comp h' f) h
    -/
  · rw [isIso_iff_bijective, Function.bijective_iff_existsUnique] at hf
    /-
      case refine_1
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      G : C
      hG : CategoryTheory.IsDetector G
      A✝ B✝ : C
      f : Quiver.Hom A✝ B✝
      hf : ∀ (b : (CategoryTheory.coyoneda.obj { unop := G }).obj B✝), ExistsUnique  …
      h : Quiver.Hom G B✝
      ⊢ ExistsUnique fun h' => Eq (CategoryTheory.CategoryStruct.comp h' f) h
    -/
    exact hf h
    /-
      🎉 no goals
    -/
  · suffices IsIso ((coyoneda.obj (op G)).map f) by
      exact @isIso_of_reflects_iso _ _ _ _ _ _ _ (coyoneda.obj (op G)) _ h
    /-
      case refine_2
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      G : C
      h : (CategoryTheory.coyoneda.obj { unop := G }).ReflectsIsomorphisms
      X Y : C
      f : Quiver.Hom X Y
      hf : ∀ (h : Quiver.Hom G Y), ExistsUnique fun h' => Eq (CategoryTheory.Categor …
      ⊢ CategoryTheory.IsIso ((CategoryTheory.coyoneda.obj { unop := G }).map f)
    -/
    rwa [isIso_iff_bijective, Function.bijective_iff_existsUnique]
    /-
      🎉 no goals
    -/


theorem isCodetector_iff_reflectsIsomorphisms_yoneda_obj (G : C) :
    IsCodetector G ↔ (yoneda.obj G).ReflectsIsomorphisms := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    G : C
    ⊢ Iff (CategoryTheory.IsCodetector G) (CategoryTheory.yoneda.obj G).ReflectsIs …
  -/
  refine ⟨fun hG => ⟨fun f hf => ?_⟩, fun h => (isCodetector_def _).2 fun X Y f hf => ?_⟩
    /-
      case refine_1
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      G : C
      hG : CategoryTheory.IsCodetector G
      A✝ B✝ : Opposite C
      f : Quiver.Hom A✝ B✝
      hf : CategoryTheory.IsIso ((CategoryTheory.yoneda.obj G).map f)
      ⊢ CategoryTheory.IsIso f
    -/
  · refine (isIso_unop_iff _).1 (hG.def _ ?_)
    /-
      case refine_1
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      G : C
      hG : CategoryTheory.IsCodetector G
      A✝ B✝ : Opposite C
      f : Quiver.Hom A✝ B✝
      hf : CategoryTheory.IsIso ((CategoryTheory.yoneda.obj G).map f)
      ⊢ ∀ (h : Quiver.Hom (Opposite.unop B✝) G), ExistsUnique fun h' => Eq (Category …
    -/
    rwa [isIso_iff_bijective, Function.bijective_iff_existsUnique] at hf
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      G : C
      h : (CategoryTheory.yoneda.obj G).ReflectsIsomorphisms
      X Y : C
      f : Quiver.Hom X Y
      hf : ∀ (h : Quiver.Hom X G), ExistsUnique fun h' => Eq (CategoryTheory.Categor …
      ⊢ CategoryTheory.IsIso f
    -/
  · rw [← isIso_op_iff]
    suffices IsIso ((yoneda.obj G).map f.op) by
      exact @isIso_of_reflects_iso _ _ _ _ _ _ _ (yoneda.obj G) _ h
    /-
      case refine_2
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      G : C
      h : (CategoryTheory.yoneda.obj G).ReflectsIsomorphisms
      X Y : C
      f : Quiver.Hom X Y
      hf : ∀ (h : Quiver.Hom X G), ExistsUnique fun h' => Eq (CategoryTheory.Categor …
      ⊢ CategoryTheory.IsIso ((CategoryTheory.yoneda.obj G).map f.op)
    -/
    rwa [isIso_iff_bijective, Function.bijective_iff_existsUnique]
    /-
      🎉 no goals
    -/


theorem wellPowered_of_isDetector [HasPullbacks C] (G : C) (hG : IsDetector G) :
    WellPowered.{v₁} C :=
  -- Porting note: added the following `haveI` to prevent universe issues
  haveI := small_subsingleton ({G} : Set C)
  wellPowered_of_isDetecting hG


theorem wellPowered_of_isSeparator [HasPullbacks C] [Balanced C] (G : C) (hG : IsSeparator G) :
    WellPowered.{v₁} C := wellPowered_of_isDetecting hG.isDetector


/--
For a category `C` and an object `G : C`, `G` is a separator of `C` if
the functor `C(G, -)` is faithful.

While `IsSeparator G : Prop` is the proposition that `G` is a separator of `C`,
an `HasSeparator C : Prop` is the proposition that such a separator exists.
Note that `HasSeparator C` is a proposition. It does not designate a favored separator
and merely asserts the existence of one.
-/
class HasSeparator : Prop where
  hasSeparator : ∃ G : C, IsSeparator G


/--
For a category `C` and an object `G : C`, `G` is a coseparator of `C` if
the functor `C(-, G)` is faithful.

While `IsCoseparator G : Prop` is the proposition that `G` is a coseparator of `C`,
an `HasCoseparator C : Prop` is the proposition that such a coseparator exists.
Note that `HasCoseparator C` is a proposition. It does not designate a favored coseparator
and merely asserts the existence of one.
-/
class HasCoseparator : Prop where
  hasCoseparator : ∃ G : C, IsCoseparator G


/--
For a category `C` and an object `G : C`, `G` is a detector of `C` if
the functor `C(G, -)` reflects isomorphisms.

While `IsDetector G : Prop` is the proposition that `G` is a detector of `C`,
an `HasDetector C : Prop` is the proposition that such a detector exists.
Note that `HasDetector C` is a proposition. It does not designate a favored detector
and merely asserts the existence of one.
-/
class HasDetector : Prop where
  hasDetector : ∃ G : C, IsDetector G


/--
For a category `C` and an object `G : C`, `G` is a codetector of `C` if
the functor `C(-, G)` reflects isomorphisms.

While `IsCodetector G : Prop` is the proposition that `G` is a codetector of `C`,
an `HasCodetector C : Prop` is the proposition that such a codetector exists.
Note that `HasCodetector C` is a proposition. It does not designate a favored codetector
and merely asserts the existence of one.
-/
class HasCodetector : Prop where
  hasCodetector : ∃ G : C, IsCodetector G


/--
Given a category `C` that has a separator (`HasSeparator C`), `separator C` is an arbitrarily
chosen separator of `C`.
-/
noncomputable def separator [HasSeparator C] : C := HasSeparator.hasSeparator.choose


/--
Given a category `C` that has a coseparator (`HasCoseparator C`), `coseparator C` is an arbitrarily
chosen coseparator of `C`.
-/
noncomputable def coseparator [HasCoseparator C] : C := HasCoseparator.hasCoseparator.choose


/--
Given a category `C` that has a detector (`HasDetector C`), `detector C` is an arbitrarily
chosen detector of `C`.
-/
noncomputable def detector [HasDetector C] : C := HasDetector.hasDetector.choose


/--
Given a category `C` that has a codetector (`HasCodetector C`), `codetector C` is an arbitrarily
chosen codetector of `C`.
-/
noncomputable def codetector [HasCodetector C] : C := HasCodetector.hasCodetector.choose


theorem isSeparator_separator [HasSeparator C] : IsSeparator (separator C) :=
  HasSeparator.hasSeparator.choose_spec


theorem isDetector_separator [Balanced C] [HasSeparator C] : IsDetector (separator C) :=
  isSeparator_separator C |>.isDetector


theorem isCoseparator_coseparator [HasCoseparator C] : IsCoseparator (coseparator C) :=
  HasCoseparator.hasCoseparator.choose_spec


theorem isCodetector_coseparator [Balanced C] [HasCoseparator C] : IsCodetector (coseparator C) :=
  isCoseparator_coseparator C |>.isCodetector


theorem isDetector_detector [HasDetector C] : IsDetector (detector C) :=
  HasDetector.hasDetector.choose_spec


theorem isSeparator_detector [HasEqualizers C] [HasDetector C] : IsSeparator (detector C) :=
  isDetector_detector C |>.isSeparator


theorem isCodetector_codetector [HasCodetector C] : IsCodetector (codetector C) :=
  HasCodetector.hasCodetector.choose_spec


theorem isCoseparator_codetector [HasCoequalizers C] [HasCodetector C] :
    IsCoseparator (codetector C) := isCodetector_codetector C |>.isCoseparator


theorem HasSeparator.hasDetector [Balanced C] [HasSeparator C] : HasDetector C :=
  ⟨_, isDetector_separator C⟩


theorem HasDetector.hasSeparator [HasEqualizers C] [HasDetector C] : HasSeparator C :=
  ⟨_, isSeparator_detector C⟩


theorem HasCoseparator.hasCodetector [Balanced C] [HasCoseparator C] : HasCodetector C :=
  ⟨_, isCodetector_coseparator C⟩


theorem HasCodetector.hasCoseparator [HasCoequalizers C] [HasCodetector C] : HasCoseparator C :=
  ⟨_, isCoseparator_codetector C⟩


instance HasDetector.wellPowered [HasPullbacks C] [HasDetector C] : WellPowered.{v₁} C :=
  isDetector_detector C |> wellPowered_of_isDetector _


instance HasSeparator.wellPowered [HasPullbacks C] [Balanced C] [HasSeparator C] :
    WellPowered.{v₁} C := HasSeparator.hasDetector.wellPowered


theorem HasSeparator.of_equivalence [HasSeparator C] (α : C ≌ D) : HasSeparator D :=
  ⟨α.functor.obj (separator C), isSeparator_separator C |>.of_equivalence α⟩


theorem HasCoseparator.of_equivalence [HasCoseparator C] (α : C ≌ D) : HasCoseparator D :=
  ⟨α.functor.obj (coseparator C), isCoseparator_coseparator C |>.of_equivalence α⟩


@[simp]
theorem hasSeparator_op_iff : HasSeparator Cᵒᵖ ↔ HasCoseparator C :=
  ⟨fun ⟨G, hG⟩ => ⟨unop G, (isCoseparator_unop_iff G).mpr hG⟩,
   fun ⟨G, hG⟩ => ⟨op G, (isSeparator_op_iff G).mpr hG⟩⟩


@[simp]
theorem hasCoseparator_op_iff : HasCoseparator Cᵒᵖ ↔ HasSeparator C :=
  ⟨fun ⟨G, hG⟩ => ⟨unop G, (isSeparator_unop_iff G).mpr hG⟩,
   fun ⟨G, hG⟩ => ⟨op G, (isCoseparator_op_iff G).mpr hG⟩⟩


@[simp]
theorem hasDetector_op_iff : HasDetector Cᵒᵖ ↔ HasCodetector C :=
  ⟨fun ⟨G, hG⟩ => ⟨unop G, (isCodetector_unop_iff G).mpr hG⟩,
   fun ⟨G, hG⟩ => ⟨op G, (isDetector_op_iff G).mpr hG⟩⟩


@[simp]
theorem hasCodetector_op_iff : HasCodetector Cᵒᵖ ↔ HasDetector C :=
  ⟨fun ⟨G, hG⟩ => ⟨unop G, (isDetector_unop_iff G).mpr hG⟩,
   fun ⟨G, hG⟩ => ⟨op G, (isCodetector_op_iff G).mpr hG⟩⟩


                                                                                    /-
                                                                                      C : Type u₁
                                                                                      inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                                                      D : Type u₂
                                                                                      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                                                                      inst✝ : CategoryTheory.HasSeparator C
                                                                                      ⊢ CategoryTheory.HasCoseparator (Opposite C)
                                                                                    -/
instance HasSeparator.hasCoseparator_op [HasSeparator C] : HasCoseparator Cᵒᵖ := by simp [*]
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/

theorem HasSeparator.hasCoseparator_of_hasSeparator_op [h : HasSeparator Cᵒᵖ] :
                           /-
                             C : Type u₁
                             inst✝ : CategoryTheory.Category.{v₁, u₁} C
                             h : CategoryTheory.HasSeparator (Opposite C)
                             ⊢ CategoryTheory.HasCoseparator C
                           -/
    HasCoseparator C := by simp_all
                           /-
                             🎉 no goals
                           -/


                                                                                    /-
                                                                                      C : Type u₁
                                                                                      inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                                                      D : Type u₂
                                                                                      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                                                                      inst✝ : CategoryTheory.HasCoseparator C
                                                                                      ⊢ CategoryTheory.HasSeparator (Opposite C)
                                                                                    -/
instance HasCoseparator.hasSeparator_op [HasCoseparator C] : HasSeparator Cᵒᵖ := by simp [*]
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/

theorem HasCoseparator.hasSeparator_of_hasCoseparator_op [HasCoseparator Cᵒᵖ] :
                         /-
                           C : Type u₁
                           inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                           inst✝ : CategoryTheory.HasCoseparator (Opposite C)
                           ⊢ CategoryTheory.HasSeparator C
                         -/
    HasSeparator C := by simp_all
                         /-
                           🎉 no goals
                         -/


                                                                                /-
                                                                                  C : Type u₁
                                                                                  inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                                                  D : Type u₂
                                                                                  inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                                                                  inst✝ : CategoryTheory.HasDetector C
                                                                                  ⊢ CategoryTheory.HasCodetector (Opposite C)
                                                                                -/
instance HasDetector.hasCodetector_op [HasDetector C] : HasCodetector Cᵒᵖ := by simp [*]
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/

theorem HasDetector.hasCodetector_of_hasDetector_op [HasDetector Cᵒᵖ] :
                          /-
                            C : Type u₁
                            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                            inst✝ : CategoryTheory.HasDetector (Opposite C)
                            ⊢ CategoryTheory.HasCodetector C
                          -/
    HasCodetector C := by simp_all
                          /-
                            🎉 no goals
                          -/


                                                                                /-
                                                                                  C : Type u₁
                                                                                  inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                                                  D : Type u₂
                                                                                  inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                                                                  inst✝ : CategoryTheory.HasCodetector C
                                                                                  ⊢ CategoryTheory.HasDetector (Opposite C)
                                                                                -/
instance HasCodetector.hasDetector_op [HasCodetector C] : HasDetector Cᵒᵖ := by simp [*]
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/

theorem HasCodetector.hasDetector_of_hasCodetector_op [HasCodetector Cᵒᵖ] :
                        /-
                          C : Type u₁
                          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                          inst✝ : CategoryTheory.HasCodetector (Opposite C)
                          ⊢ CategoryTheory.HasDetector C
                        -/
    HasDetector C := by simp_all
                        /-
                          🎉 no goals
                        -/


