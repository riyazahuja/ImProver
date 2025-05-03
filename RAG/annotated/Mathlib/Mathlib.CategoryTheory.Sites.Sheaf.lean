/-- A sheaf of A is a presheaf P : Cᵒᵖ => A such that for every E : A, the
presheaf of types given by sending U : C to Hom_{A}(E, P U) is a sheaf of types.

https://stacks.math.columbia.edu/tag/00VR
-/
def IsSheaf (P : Cᵒᵖ ⥤ A) : Prop :=
  ∀ E : A, Presieve.IsSheaf J (P ⋙ coyoneda.obj (op E))


attribute [local instance] ConcreteCategory.hasCoeToSort ConcreteCategory.instFunLike in
/-- Condition that a presheaf with values in a concrete category is separated for
a Grothendieck topology. -/
def IsSeparated (P : Cᵒᵖ ⥤ A) [ConcreteCategory A] : Prop :=
  ∀ (X : C) (S : Sieve X) (_ : S ∈ J X) (x y : P.obj (op X)),
    (∀ (Y : C) (f : Y ⟶ X) (_ : S f), P.map f.op x = P.map f.op y) → x = y


/-- Given a sieve `S` on `X : C`, a presheaf `P : Cᵒᵖ ⥤ A`, and an object `E` of `A`,
    the cones over the natural diagram `S.arrows.diagram.op ⋙ P` associated to `S` and `P`
    with cone point `E` are in 1-1 correspondence with sieve_compatible family of elements
    for the sieve `S` and the presheaf of types `Hom (E, P -)`. -/
@[simps]
def conesEquivSieveCompatibleFamily :
    (S.arrows.diagram.op ⋙ P).cones.obj E ≃
      { x : FamilyOfElements (P ⋙ coyoneda.obj E) (S : Presieve X) // x.SieveCompatible } where
  toFun π :=
    ⟨fun _ f h => π.app (op ⟨Over.mk f, h⟩), fun X Y f g hf => by
      /-
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        A : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} A
        J : CategoryTheory.GrothendieckTopology C
        P : CategoryTheory.Functor (Opposite C) A
        X✝ : C
        S : CategoryTheory.Sieve X✝
        R : CategoryTheory.Presieve X✝
        E : Opposite A
        π : (S.arrows.diagram.op.comp P).cones.obj E
        X Y : C
        f : Quiver.Hom X X✝
        g : Quiver.Hom Y X
        hf : S.arrows f
        ⊢ Eq ((fun x f h => π.app { unop := { obj := CategoryTheory.Over.mk f, propert …
      -/
      apply (id_comp _).symm.trans
      /-
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        A : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} A
        J : CategoryTheory.GrothendieckTopology C
        P : CategoryTheory.Functor (Opposite C) A
        X✝ : C
        S : CategoryTheory.Sieve X✝
        R : CategoryTheory.Presieve X✝
        E : Opposite A
        π : (S.arrows.diagram.op.comp P).cones.obj E
        X Y : C
        f : Quiver.Hom X X✝
        g : Quiver.Hom Y X
        hf : S.arrows f
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ((O …
      -/
      dsimp
      /-
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        A : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} A
        J : CategoryTheory.GrothendieckTopology C
        P : CategoryTheory.Functor (Opposite C) A
        X✝ : C
        S : CategoryTheory.Sieve X✝
        R : CategoryTheory.Presieve X✝
        E : Opposite A
        π : (S.arrows.diagram.op.comp P).cones.obj E
        X Y : C
        f : Quiver.Hom X X✝
        g : Quiver.Hom Y X
        hf : S.arrows f
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (Op …
      -/
      exact π.naturality (Quiver.Hom.op (Over.homMk _ (by rfl)))⟩
      /-
        🎉 no goals
      -/
  invFun x :=
    { app := fun f => x.1 f.unop.1.hom f.unop.2
      naturality := fun f f' g => by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          A : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} A
          J : CategoryTheory.GrothendieckTopology C
          P : CategoryTheory.Functor (Opposite C) A
          X : C
          S : CategoryTheory.Sieve X
          R : CategoryTheory.Presieve X
          E : Opposite A
          x : Subtype fun x => x.SieveCompatible
          f f' : Opposite S.arrows.category
          g : Quiver.Hom f f'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Opposite.unop ((CategoryTheory.Func …
        -/
        refine Eq.trans ?_ (x.2 f.unop.1.hom g.unop.left f.unop.2)
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          A : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} A
          J : CategoryTheory.GrothendieckTopology C
          P : CategoryTheory.Functor (Opposite C) A
          X : C
          S : CategoryTheory.Sieve X
          R : CategoryTheory.Presieve X
          E : Opposite A
          x : Subtype fun x => x.SieveCompatible
          f f' : Opposite S.arrows.category
          g : Quiver.Hom f f'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Opposite.unop ((CategoryTheory.Func …
        -/
        dsimp
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          A : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} A
          J : CategoryTheory.GrothendieckTopology C
          P : CategoryTheory.Functor (Opposite C) A
          X : C
          S : CategoryTheory.Sieve X
          R : CategoryTheory.Presieve X
          E : Opposite A
          x : Subtype fun x => x.SieveCompatible
          f f' : Opposite S.arrows.category
          g : Quiver.Hom f f'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (Op …
        -/
        rw [id_comp]
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          A : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} A
          J : CategoryTheory.GrothendieckTopology C
          P : CategoryTheory.Functor (Opposite C) A
          X : C
          S : CategoryTheory.Sieve X
          R : CategoryTheory.Presieve X
          E : Opposite A
          x : Subtype fun x => x.SieveCompatible
          f f' : Opposite S.arrows.category
          g : Quiver.Hom f f'
          ⊢ Eq (↑x (Opposite.unop f').obj.hom ⋯) (↑x (CategoryTheory.CategoryStruct.comp …
        -/
        convert rfl
        /-
          case h.e'_3.h.e'_4
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          A : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} A
          J : CategoryTheory.GrothendieckTopology C
          P : CategoryTheory.Functor (Opposite C) A
          X : C
          S : CategoryTheory.Sieve X
          R : CategoryTheory.Presieve X
          E : Opposite A
          x : Subtype fun x => x.SieveCompatible
          f f' : Opposite S.arrows.category
          g : Quiver.Hom f f'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp g.unop.left (Opposite.unop f).obj.hom …
        -/
        rw [Over.w] }
        /-
          🎉 no goals
        -/
  left_inv _ := rfl
  right_inv _ := rfl

-- These lemmas have always been bad (https://github.com/leanprover-community/mathlib4/issues/7657), but https://github.com/leanprover/lean4/pull/2644 made `simp` start noticing

/-- The cone corresponding to a sieve_compatible family of elements, dot notation enabled. -/
@[simp]
def _root_.CategoryTheory.Presieve.FamilyOfElements.SieveCompatible.cone :
    Cone (S.arrows.diagram.op ⋙ P) where
  pt := E.unop
  π := (conesEquivSieveCompatibleFamily P S E).invFun ⟨x, hx⟩


/-- Cone morphisms from the cone corresponding to a sieve_compatible family to the natural
    cone associated to a sieve `S` and a presheaf `P` are in 1-1 correspondence with amalgamations
    of the family. -/
def homEquivAmalgamation :
    (hx.cone ⟶ P.mapCone S.arrows.cocone.op) ≃ { t // x.IsAmalgamation t } where
  toFun l := ⟨l.hom, fun _ f hf => l.w (op ⟨Over.mk f, hf⟩)⟩
  invFun t := ⟨t.1, fun f => t.2 f.unop.1.hom f.unop.2⟩
  left_inv _ := rfl
  right_inv _ := rfl


/-- Given sieve `S` and presheaf `P : Cᵒᵖ ⥤ A`, their natural associated cone is a limit cone
    iff `Hom (E, P -)` is a sheaf of types for the sieve `S` and all `E : A`. -/
theorem isLimit_iff_isSheafFor :
    Nonempty (IsLimit (P.mapCone S.arrows.cocone.op)) ↔
      ∀ E : Aᵒᵖ, IsSheafFor (P ⋙ coyoneda.obj E) S.arrows := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} A
    P : CategoryTheory.Functor (Opposite C) A
    X : C
    S : CategoryTheory.Sieve X
    ⊢ Iff (Nonempty (CategoryTheory.Limits.IsLimit (P.mapCone S.arrows.cocone.op)) …
  -/
  dsimp [IsSheafFor]; simp_rw [compatible_iff_sieveCompatible]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} A
    P : CategoryTheory.Functor (Opposite C) A
    X : C
    S : CategoryTheory.Sieve X
    ⊢ Iff (Nonempty (CategoryTheory.Limits.IsLimit (P.mapCone S.arrows.cocone.op)) …
  -/
  rw [((Cone.isLimitEquivIsTerminal _).trans (isTerminalEquivUnique _ _)).nonempty_congr]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} A
    P : CategoryTheory.Functor (Opposite C) A
    X : C
    S : CategoryTheory.Sieve X
    ⊢ Iff (Nonempty ((X_1 : CategoryTheory.Limits.Cone (S.arrows.diagram.op.comp P …
  -/
  rw [Classical.nonempty_pi]; constructor
    /-
      case mp
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} A
      P : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      ⊢ (∀ (i : CategoryTheory.Limits.Cone (S.arrows.diagram.op.comp P)), Nonempty ( …
    -/
  · intro hu E x hx
    /-
      case mp
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} A
      P : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      hu : ∀ (i : CategoryTheory.Limits.Cone (S.arrows.diagram.op.comp P)), Nonempty …
      E : Opposite A
      x : CategoryTheory.Presieve.FamilyOfElements (P.comp (CategoryTheory.coyoneda. …
      hx : x.SieveCompatible
      ⊢ ExistsUnique fun t => x.IsAmalgamation t
    -/
    specialize hu hx.cone
    /-
      case mp
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} A
      P : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      E : Opposite A
      x : CategoryTheory.Presieve.FamilyOfElements (P.comp (CategoryTheory.coyoneda. …
      hx : x.SieveCompatible
      hu : Nonempty (Unique (Quiver.Hom hx.cone (P.mapCone S.arrows.cocone.op)))
      ⊢ ExistsUnique fun t => x.IsAmalgamation t
    -/
    rw [(homEquivAmalgamation hx).uniqueCongr.nonempty_congr] at hu
    /-
      case mp
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} A
      P : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      E : Opposite A
      x : CategoryTheory.Presieve.FamilyOfElements (P.comp (CategoryTheory.coyoneda. …
      hx : x.SieveCompatible
      hu : Nonempty (Unique (Subtype fun t => x.IsAmalgamation t))
      ⊢ ExistsUnique fun t => x.IsAmalgamation t
    -/
    exact (unique_subtype_iff_existsUnique _).1 hu
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} A
      P : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      ⊢ (∀ (E : Opposite A) (x : CategoryTheory.Presieve.FamilyOfElements (P.comp (C …
    -/
  · rintro h ⟨E, π⟩
    /-
      case mpr.mk
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} A
      P : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      h : ∀ (E : Opposite A) (x : CategoryTheory.Presieve.FamilyOfElements (P.comp ( …
      E : A
      π : Quiver.Hom ((CategoryTheory.Functor.const (Opposite S.arrows.category)).ob …
      ⊢ Nonempty (Unique (Quiver.Hom { pt := E, π := π } (P.mapCone S.arrows.cocone. …
    -/
    let eqv := conesEquivSieveCompatibleFamily P S (op E)
    /-
      case mpr.mk
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} A
      P : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      h : ∀ (E : Opposite A) (x : CategoryTheory.Presieve.FamilyOfElements (P.comp ( …
      E : A
      π : Quiver.Hom ((CategoryTheory.Functor.const (Opposite S.arrows.category)).ob …
      eqv : Equiv ((S.arrows.diagram.op.comp P).cones.obj { unop := E }) (Subtype fu …
      ⊢ Nonempty (Unique (Quiver.Hom { pt := E, π := π } (P.mapCone S.arrows.cocone. …
    -/
    rw [← eqv.left_inv π]
    /-
      case mpr.mk
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} A
      P : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      h : ∀ (E : Opposite A) (x : CategoryTheory.Presieve.FamilyOfElements (P.comp ( …
      E : A
      π : Quiver.Hom ((CategoryTheory.Functor.const (Opposite S.arrows.category)).ob …
      eqv : Equiv ((S.arrows.diagram.op.comp P).cones.obj { unop := E }) (Subtype fu …
      ⊢ Nonempty (Unique (Quiver.Hom { pt := E, π := eqv.invFun (eqv.toFun π) } (P.m …
    -/
    erw [(homEquivAmalgamation (eqv π).2).uniqueCongr.nonempty_congr]
    /-
      case mpr.mk
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} A
      P : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      h : ∀ (E : Opposite A) (x : CategoryTheory.Presieve.FamilyOfElements (P.comp ( …
      E : A
      π : Quiver.Hom ((CategoryTheory.Functor.const (Opposite S.arrows.category)).ob …
      eqv : Equiv ((S.arrows.diagram.op.comp P).cones.obj { unop := E }) (Subtype fu …
      ⊢ Nonempty (Unique (Subtype fun t => (↑(eqv π)).IsAmalgamation t))
    -/
    rw [unique_subtype_iff_existsUnique]
    /-
      case mpr.mk
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} A
      P : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      h : ∀ (E : Opposite A) (x : CategoryTheory.Presieve.FamilyOfElements (P.comp ( …
      E : A
      π : Quiver.Hom ((CategoryTheory.Functor.const (Opposite S.arrows.category)).ob …
      eqv : Equiv ((S.arrows.diagram.op.comp P).cones.obj { unop := E }) (Subtype fu …
      ⊢ ExistsUnique fun a => (↑(eqv π)).IsAmalgamation a
    -/
    exact h _ _ (eqv π).2
    /-
      🎉 no goals
    -/


/-- Given sieve `S` and presheaf `P : Cᵒᵖ ⥤ A`, their natural associated cone admits at most one
    morphism from every cone in the same category (i.e. over the same diagram),
    iff `Hom (E, P -)`is separated for the sieve `S` and all `E : A`. -/
theorem subsingleton_iff_isSeparatedFor :
    (∀ c, Subsingleton (c ⟶ P.mapCone S.arrows.cocone.op)) ↔
      ∀ E : Aᵒᵖ, IsSeparatedFor (P ⋙ coyoneda.obj E) S.arrows := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} A
    P : CategoryTheory.Functor (Opposite C) A
    X : C
    S : CategoryTheory.Sieve X
    ⊢ Iff (∀ (c : CategoryTheory.Limits.Cone (S.arrows.diagram.op.comp P)), Subsin …
  -/
  constructor
    /-
      case mp
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} A
      P : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      ⊢ (∀ (c : CategoryTheory.Limits.Cone (S.arrows.diagram.op.comp P)), Subsinglet …
    -/
  · intro hs E x t₁ t₂ h₁ h₂
    /-
      case mp
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} A
      P : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      hs : ∀ (c : CategoryTheory.Limits.Cone (S.arrows.diagram.op.comp P)), Subsingl …
      E : Opposite A
      x : CategoryTheory.Presieve.FamilyOfElements (P.comp (CategoryTheory.coyoneda. …
      t₁ t₂ : (P.comp (CategoryTheory.coyoneda.obj E)).obj { unop := X }
      h₁ : x.IsAmalgamation t₁
      h₂ : x.IsAmalgamation t₂
      ⊢ Eq t₁ t₂
    -/
    have hx := is_compatible_of_exists_amalgamation x ⟨t₁, h₁⟩
    /-
      case mp
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} A
      P : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      hs : ∀ (c : CategoryTheory.Limits.Cone (S.arrows.diagram.op.comp P)), Subsingl …
      E : Opposite A
      x : CategoryTheory.Presieve.FamilyOfElements (P.comp (CategoryTheory.coyoneda. …
      t₁ t₂ : (P.comp (CategoryTheory.coyoneda.obj E)).obj { unop := X }
      h₁ : x.IsAmalgamation t₁
      h₂ : x.IsAmalgamation t₂
      hx : x.Compatible
      ⊢ Eq t₁ t₂
    -/
    rw [compatible_iff_sieveCompatible] at hx
    /-
      case mp
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} A
      P : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      hs : ∀ (c : CategoryTheory.Limits.Cone (S.arrows.diagram.op.comp P)), Subsingl …
      E : Opposite A
      x : CategoryTheory.Presieve.FamilyOfElements (P.comp (CategoryTheory.coyoneda. …
      t₁ t₂ : (P.comp (CategoryTheory.coyoneda.obj E)).obj { unop := X }
      h₁ : x.IsAmalgamation t₁
      h₂ : x.IsAmalgamation t₂
      hx : x.SieveCompatible
      ⊢ Eq t₁ t₂
    -/
    specialize hs hx.cone
    /-
      case mp
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} A
      P : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      E : Opposite A
      x : CategoryTheory.Presieve.FamilyOfElements (P.comp (CategoryTheory.coyoneda. …
      t₁ t₂ : (P.comp (CategoryTheory.coyoneda.obj E)).obj { unop := X }
      h₁ : x.IsAmalgamation t₁
      h₂ : x.IsAmalgamation t₂
      hx : x.SieveCompatible
      hs : Subsingleton (Quiver.Hom hx.cone (P.mapCone S.arrows.cocone.op))
      ⊢ Eq t₁ t₂
    -/
    rcases hs with ⟨hs⟩
    simpa only [Subtype.mk.injEq] using (show Subtype.mk t₁ h₁ = ⟨t₂, h₂⟩ from
      (homEquivAmalgamation hx).symm.injective (hs _ _))
    /-
      case mpr
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} A
      P : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      ⊢ (∀ (E : Opposite A), CategoryTheory.Presieve.IsSeparatedFor (P.comp (Categor …
    -/
  · rintro h ⟨E, π⟩
    /-
      case mpr.mk
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} A
      P : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      h : ∀ (E : Opposite A), CategoryTheory.Presieve.IsSeparatedFor (P.comp (Catego …
      E : A
      π : Quiver.Hom ((CategoryTheory.Functor.const (Opposite S.arrows.category)).ob …
      ⊢ Subsingleton (Quiver.Hom { pt := E, π := π } (P.mapCone S.arrows.cocone.op))
    -/
    let eqv := conesEquivSieveCompatibleFamily P S (op E)
    /-
      case mpr.mk
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} A
      P : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      h : ∀ (E : Opposite A), CategoryTheory.Presieve.IsSeparatedFor (P.comp (Catego …
      E : A
      π : Quiver.Hom ((CategoryTheory.Functor.const (Opposite S.arrows.category)).ob …
      eqv : Equiv ((S.arrows.diagram.op.comp P).cones.obj { unop := E }) (Subtype fu …
      ⊢ Subsingleton (Quiver.Hom { pt := E, π := π } (P.mapCone S.arrows.cocone.op))
    -/
    constructor
    /-
      case mpr.mk.allEq
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} A
      P : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      h : ∀ (E : Opposite A), CategoryTheory.Presieve.IsSeparatedFor (P.comp (Catego …
      E : A
      π : Quiver.Hom ((CategoryTheory.Functor.const (Opposite S.arrows.category)).ob …
      eqv : Equiv ((S.arrows.diagram.op.comp P).cones.obj { unop := E }) (Subtype fu …
      ⊢ ∀ (a b : Quiver.Hom { pt := E, π := π } (P.mapCone S.arrows.cocone.op)), Eq  …
    -/
    rw [← eqv.left_inv π]
    /-
      case mpr.mk.allEq
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} A
      P : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      h : ∀ (E : Opposite A), CategoryTheory.Presieve.IsSeparatedFor (P.comp (Catego …
      E : A
      π : Quiver.Hom ((CategoryTheory.Functor.const (Opposite S.arrows.category)).ob …
      eqv : Equiv ((S.arrows.diagram.op.comp P).cones.obj { unop := E }) (Subtype fu …
      ⊢ ∀ (a b : Quiver.Hom { pt := E, π := eqv.invFun (eqv.toFun π) } (P.mapCone S. …
    -/
    intro f₁ f₂
    /-
      case mpr.mk.allEq
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} A
      P : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      h : ∀ (E : Opposite A), CategoryTheory.Presieve.IsSeparatedFor (P.comp (Catego …
      E : A
      π : Quiver.Hom ((CategoryTheory.Functor.const (Opposite S.arrows.category)).ob …
      eqv : Equiv ((S.arrows.diagram.op.comp P).cones.obj { unop := E }) (Subtype fu …
      f₁ f₂ : Quiver.Hom { pt := E, π := eqv.invFun (eqv.toFun π) } (P.mapCone S.arr …
      ⊢ Eq f₁ f₂
    -/
    let eqv' := homEquivAmalgamation (eqv π).2
    /-
      case mpr.mk.allEq
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} A
      P : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      h : ∀ (E : Opposite A), CategoryTheory.Presieve.IsSeparatedFor (P.comp (Catego …
      E : A
      π : Quiver.Hom ((CategoryTheory.Functor.const (Opposite S.arrows.category)).ob …
      eqv : Equiv ((S.arrows.diagram.op.comp P).cones.obj { unop := E }) (Subtype fu …
      f₁ f₂ : Quiver.Hom { pt := E, π := eqv.invFun (eqv.toFun π) } (P.mapCone S.arr …
      eqv' : Equiv (Quiver.Hom ⋯.cone (P.mapCone S.arrows.cocone.op)) (Subtype fun t …
      ⊢ Eq f₁ f₂
    -/
    apply eqv'.injective
    /-
      case mpr.mk.allEq.a
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} A
      P : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      h : ∀ (E : Opposite A), CategoryTheory.Presieve.IsSeparatedFor (P.comp (Catego …
      E : A
      π : Quiver.Hom ((CategoryTheory.Functor.const (Opposite S.arrows.category)).ob …
      eqv : Equiv ((S.arrows.diagram.op.comp P).cones.obj { unop := E }) (Subtype fu …
      f₁ f₂ : Quiver.Hom { pt := E, π := eqv.invFun (eqv.toFun π) } (P.mapCone S.arr …
      eqv' : Equiv (Quiver.Hom ⋯.cone (P.mapCone S.arrows.cocone.op)) (Subtype fun t …
      ⊢ Eq (eqv' f₁) (eqv' f₂)
    -/
    ext
    /-
      case mpr.mk.allEq.a.a
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} A
      P : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      h : ∀ (E : Opposite A), CategoryTheory.Presieve.IsSeparatedFor (P.comp (Catego …
      E : A
      π : Quiver.Hom ((CategoryTheory.Functor.const (Opposite S.arrows.category)).ob …
      eqv : Equiv ((S.arrows.diagram.op.comp P).cones.obj { unop := E }) (Subtype fu …
      f₁ f₂ : Quiver.Hom { pt := E, π := eqv.invFun (eqv.toFun π) } (P.mapCone S.arr …
      eqv' : Equiv (Quiver.Hom ⋯.cone (P.mapCone S.arrows.cocone.op)) (Subtype fun t …
      ⊢ Eq ↑(eqv' f₁) ↑(eqv' f₂)
    -/
                            /-
                              🎉 no goals
                            -/
    apply h _ (eqv π).1 <;> exact (eqv' _).2
                            /-
                              🎉 no goals
                            -/


/-- A presheaf `P` is a sheaf for the Grothendieck topology `J` iff for every covering sieve
    `S` of `J`, the natural cone associated to `P` and `S` is a limit cone. -/
theorem isSheaf_iff_isLimit :
    IsSheaf J P ↔
      ∀ ⦃X : C⦄ (S : Sieve X), S ∈ J X → Nonempty (IsLimit (P.mapCone S.arrows.cocone.op)) :=
  ⟨fun h _ S hS => (isLimit_iff_isSheafFor P S).2 fun E => h E.unop S hS, fun h E _ S hS =>
    (isLimit_iff_isSheafFor P S).1 (h S hS) (op E)⟩


/-- A presheaf `P` is separated for the Grothendieck topology `J` iff for every covering sieve
    `S` of `J`, the natural cone associated to `P` and `S` admits at most one morphism from every
    cone in the same category. -/
theorem isSeparated_iff_subsingleton :
    (∀ E : A, Presieve.IsSeparated J (P ⋙ coyoneda.obj (op E))) ↔
      ∀ ⦃X : C⦄ (S : Sieve X), S ∈ J X → ∀ c, Subsingleton (c ⟶ P.mapCone S.arrows.cocone.op) :=
  ⟨fun h _ S hS => (subsingleton_iff_isSeparatedFor P S).2 fun E => h E.unop S hS, fun h E _ S hS =>
    (subsingleton_iff_isSeparatedFor P S).1 (h S hS) (op E)⟩


/-- Given presieve `R` and presheaf `P : Cᵒᵖ ⥤ A`, the natural cone associated to `P` and
    the sieve `Sieve.generate R` generated by `R` is a limit cone iff `Hom (E, P -)` is a
    sheaf of types for the presieve `R` and all `E : A`. -/
theorem isLimit_iff_isSheafFor_presieve :
    Nonempty (IsLimit (P.mapCone (generate R).arrows.cocone.op)) ↔
      ∀ E : Aᵒᵖ, IsSheafFor (P ⋙ coyoneda.obj E) R :=
  (isLimit_iff_isSheafFor P _).trans (forall_congr' fun _ => (isSheafFor_iff_generate _).symm)


/-- A presheaf `P` is a sheaf for the Grothendieck topology generated by a pretopology `K`
    iff for every covering presieve `R` of `K`, the natural cone associated to `P` and
    `Sieve.generate R` is a limit cone. -/
theorem isSheaf_iff_isLimit_pretopology [HasPullbacks C] (K : Pretopology C) :
    IsSheaf (K.toGrothendieck C) P ↔
      ∀ ⦃X : C⦄ (R : Presieve X),
        R ∈ K X → Nonempty (IsLimit (P.mapCone (generate R).arrows.cocone.op)) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
    P : CategoryTheory.Functor (Opposite C) A
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    K : CategoryTheory.Pretopology C
    ⊢ Iff (CategoryTheory.Presheaf.IsSheaf (CategoryTheory.Pretopology.toGrothendi …
  -/
  dsimp [IsSheaf]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
    P : CategoryTheory.Functor (Opposite C) A
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    K : CategoryTheory.Pretopology C
    ⊢ Iff (∀ (E : A), CategoryTheory.Presieve.IsSheaf (CategoryTheory.Pretopology. …
  -/
  simp_rw [isSheaf_pretopology]
  exact
    ⟨fun h X R hR => (isLimit_iff_isSheafFor_presieve P R).2 fun E => h E.unop R hR,
      fun h E X R hR => (isLimit_iff_isSheafFor_presieve P R).1 (h R hR) (op E)⟩


/-- This is a wrapper around `Presieve.IsSheafFor.amalgamate` to be used below.
  If `P`s a sheaf, `S` is a cover of `X`, and `x` is a collection of morphisms from `E`
  to `P` evaluated at terms in the cover which are compatible, then we can amalgamate
  the `x`s to obtain a single morphism `E ⟶ P.obj (op X)`. -/
def IsSheaf.amalgamate {A : Type u₂} [Category.{v₂} A] {E : A} {X : C} {P : Cᵒᵖ ⥤ A}
    (hP : Presheaf.IsSheaf J P) (S : J.Cover X) (x : ∀ I : S.Arrow, E ⟶ P.obj (op I.Y))
    (hx : ∀ ⦃I₁ I₂ : S.Arrow⦄ (r : I₁.Relation I₂),
       x I₁ ≫ P.map r.g₁.op = x I₂ ≫ P.map r.g₂.op) : E ⟶ P.obj (op X) :=
  (hP _ _ S.condition).amalgamate (fun Y f hf => x ⟨Y, f, hf⟩) fun _ _ _ _ _ _ _ h₁ h₂ w =>
    @hx { hf := h₁ } { hf := h₂ } { w := w }


@[reassoc (attr := simp)]
theorem IsSheaf.amalgamate_map {A : Type u₂} [Category.{v₂} A] {E : A} {X : C} {P : Cᵒᵖ ⥤ A}
    (hP : Presheaf.IsSheaf J P) (S : J.Cover X) (x : ∀ I : S.Arrow, E ⟶ P.obj (op I.Y))
    (hx : ∀ ⦃I₁ I₂ : S.Arrow⦄ (r : I₁.Relation I₂),
       x I₁ ≫ P.map r.g₁.op = x I₂ ≫ P.map r.g₂.op)
    (I : S.Arrow) :
    hP.amalgamate S x hx ≫ P.map I.f.op = x _ := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} A
    E : A
    X : C
    P : CategoryTheory.Functor (Opposite C) A
    hP : CategoryTheory.Presheaf.IsSheaf J P
    S : J.Cover X
    x : (I : S.Arrow) → Quiver.Hom E (P.obj { unop := I.Y })
    hx : ∀ ⦃I₁ I₂ : S.Arrow⦄ (r : I₁.Relation I₂), Eq (CategoryTheory.CategoryStru …
    I : S.Arrow
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (hP.amalgamate S x hx) (P.map I.f.op) …
  -/
  apply (hP _ _ S.condition).valid_glue
  /-
    🎉 no goals
  -/


theorem IsSheaf.hom_ext {A : Type u₂} [Category.{v₂} A] {E : A} {X : C} {P : Cᵒᵖ ⥤ A}
    (hP : Presheaf.IsSheaf J P) (S : J.Cover X) (e₁ e₂ : E ⟶ P.obj (op X))
    (h : ∀ I : S.Arrow, e₁ ≫ P.map I.f.op = e₂ ≫ P.map I.f.op) : e₁ = e₂ :=
  (hP _ _ S.condition).isSeparatedFor.ext fun Y f hf => h ⟨Y, f, hf⟩


lemma IsSheaf.hom_ext_ofArrows
    {P : Cᵒᵖ ⥤ A} (hP : Presheaf.IsSheaf J P) {I : Type*} {S : C} {X : I → C}
    (f : ∀ i, X i ⟶ S) (hf : Sieve.ofArrows _ f ∈ J S) {E : A}
    {x y : E ⟶ P.obj (op S)} (h : ∀ i, x ≫ P.map (f i).op = y ≫ P.map (f i).op) :
    x = y := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} A
    J : CategoryTheory.GrothendieckTopology C
    P : CategoryTheory.Functor (Opposite C) A
    hP : CategoryTheory.Presheaf.IsSheaf J P
    I : Type u_1
    S : C
    X : I → C
    f : (i : I) → Quiver.Hom (X i) S
    hf : Membership.mem (J S) (CategoryTheory.Sieve.ofArrows X f)
    E : A
    x y : Quiver.Hom E (P.obj { unop := S })
    h : ∀ (i : I), Eq (CategoryTheory.CategoryStruct.comp x (P.map (f i).op)) (Cat …
    ⊢ Eq x y
  -/
  apply hP.hom_ext ⟨_, hf⟩
  /-
    case h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} A
    J : CategoryTheory.GrothendieckTopology C
    P : CategoryTheory.Functor (Opposite C) A
    hP : CategoryTheory.Presheaf.IsSheaf J P
    I : Type u_1
    S : C
    X : I → C
    f : (i : I) → Quiver.Hom (X i) S
    hf : Membership.mem (J S) (CategoryTheory.Sieve.ofArrows X f)
    E : A
    x y : Quiver.Hom E (P.obj { unop := S })
    h : ∀ (i : I), Eq (CategoryTheory.CategoryStruct.comp x (P.map (f i).op)) (Cat …
    ⊢ ∀ (I_1 : CategoryTheory.GrothendieckTopology.Cover.Arrow ⟨CategoryTheory.Sie …
  -/
  rintro ⟨Z, _, _, g, _, ⟨i⟩, rfl⟩
  /-
    case h.mk.intro.intro.intro.intro.mk
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} A
    J : CategoryTheory.GrothendieckTopology C
    P : CategoryTheory.Functor (Opposite C) A
    hP : CategoryTheory.Presheaf.IsSheaf J P
    I : Type u_1
    S : C
    X : I → C
    f : (i : I) → Quiver.Hom (X i) S
    hf : Membership.mem (J S) (CategoryTheory.Sieve.ofArrows X f)
    E : A
    x y : Quiver.Hom E (P.obj { unop := S })
    h : ∀ (i : I), Eq (CategoryTheory.CategoryStruct.comp x (P.map (f i).op)) (Cat …
    Z Y : C
    i : I
    g : Quiver.Hom Z (X i)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp x (P.map { Y := Z, f := CategoryTheor …
  -/
  dsimp
  /-
    case h.mk.intro.intro.intro.intro.mk
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} A
    J : CategoryTheory.GrothendieckTopology C
    P : CategoryTheory.Functor (Opposite C) A
    hP : CategoryTheory.Presheaf.IsSheaf J P
    I : Type u_1
    S : C
    X : I → C
    f : (i : I) → Quiver.Hom (X i) S
    hf : Membership.mem (J S) (CategoryTheory.Sieve.ofArrows X f)
    E : A
    x y : Quiver.Hom E (P.obj { unop := S })
    h : ∀ (i : I), Eq (CategoryTheory.CategoryStruct.comp x (P.map (f i).op)) (Cat …
    Z Y : C
    i : I
    g : Quiver.Hom Z (X i)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp x (P.map (CategoryTheory.CategoryStru …
  -/
  rw [P.map_comp, reassoc_of% (h i)]
  /-
    🎉 no goals
  -/


lemma IsSheaf.existsUnique_amalgamation_ofArrows :
    ∃! (g : E ⟶ P.obj (op S)), ∀ (i : I), g ≫ P.map (f i).op = x i :=
  (Presieve.isSheafFor_arrows_iff _ _).1
    ((Presieve.isSheafFor_iff_generate _).2 (hP E _ hf)) x (fun _ _ _ _ _ w => hx _ _ w)


@[deprecated (since := "2024-12-17")]
alias IsSheaf.exists_unique_amalgamation_ofArrows := IsSheaf.existsUnique_amalgamation_ofArrows


/-- If `P : Cᵒᵖ ⥤ A` is a sheaf and `f i : X i ⟶ S` is a covering family, then
a morphism `E ⟶ P.obj (op S)` can be constructed from a compatible family of
morphisms `x : E ⟶ P.obj (op (X i))`. -/
def IsSheaf.amalgamateOfArrows : E ⟶ P.obj (op S) :=
  (hP.existsUnique_amalgamation_ofArrows f hf x hx).choose


@[reassoc (attr := simp)]
lemma IsSheaf.amalgamateOfArrows_map (i : I) :
    hP.amalgamateOfArrows f hf x hx ≫ P.map (f i).op = x i :=
  (hP.existsUnique_amalgamation_ofArrows f hf x hx).choose_spec.1 i


theorem isSheaf_of_iso_iff {P P' : Cᵒᵖ ⥤ A} (e : P ≅ P') : IsSheaf J P ↔ IsSheaf J P' :=
  forall_congr' fun _ =>
    ⟨Presieve.isSheaf_iso J (isoWhiskerRight e _),
      Presieve.isSheaf_iso J (isoWhiskerRight e.symm _)⟩


theorem isSheaf_of_isTerminal {X : A} (hX : IsTerminal X) :
    Presheaf.IsSheaf J ((CategoryTheory.Functor.const _).obj X) := fun _ _ _ _ _ _ =>
  ⟨hX.from _, fun _ _ _ => hX.hom_ext _ _, fun _ _ => hX.hom_ext _ _⟩


/-- The category of sheaves taking values in `A` on a grothendieck topology. -/
structure Sheaf where
  /-- the underlying presheaf -/
  val : Cᵒᵖ ⥤ A
  /-- the condition that the presheaf is a sheaf -/
  cond : Presheaf.IsSheaf J val


/-- Morphisms between sheaves are just morphisms of presheaves. -/
@[ext]
structure Hom (X Y : Sheaf J A) where
  /-- a morphism between the underlying presheaves -/
  val : X.val ⟶ Y.val


@[simps id_val comp_val]
instance instCategorySheaf : Category (Sheaf J A) where
  Hom := Hom
  id _ := ⟨𝟙 _⟩
  comp f g := ⟨f.val ≫ g.val⟩
  id_comp _ := Hom.ext <| id_comp _
  comp_id _ := Hom.ext <| comp_id _
  assoc _ _ _ := Hom.ext <| assoc _ _ _

-- Let's make the inhabited linter happy.../sips

instance (X : Sheaf J A) : Inhabited (Hom X X) :=
  ⟨𝟙 X⟩


@[ext]
lemma hom_ext {X Y : Sheaf J A} (x y : X ⟶ Y) (h : x.val = y.val) : x = y :=
  Sheaf.Hom.ext h


/-- The inclusion functor from sheaves to presheaves. -/
@[simps]
def sheafToPresheaf : Sheaf J A ⥤ Cᵒᵖ ⥤ A where
  obj := Sheaf.val
  map f := f.val
  map_id _ := rfl
  map_comp _ _ := rfl


/-- The sections of a sheaf (i.e. evaluation as a presheaf on `C`). -/
abbrev sheafSections : Cᵒᵖ ⥤ Sheaf J A ⥤ A := (sheafToPresheaf J A).flip


/-- The functor `Sheaf J A ⥤ Cᵒᵖ ⥤ A` is fully faithful. -/
@[simps]
def fullyFaithfulSheafToPresheaf : (sheafToPresheaf J A).FullyFaithful where
  preimage f := ⟨f⟩


variable {J A} in
/-- The bijection `(X ⟶ Y) ≃ (X.val ⟶ Y.val)` when `X` and `Y` are sheaves. -/
abbrev Sheaf.homEquiv {X Y : Sheaf J A} : (X ⟶ Y) ≃ (X.val ⟶ Y.val) :=
  (fullyFaithfulSheafToPresheaf J A).homEquiv


instance : (sheafToPresheaf J A).Full :=
  (fullyFaithfulSheafToPresheaf J A).full


instance : (sheafToPresheaf J A).Faithful :=
  (fullyFaithfulSheafToPresheaf J A).faithful


instance : (sheafToPresheaf J A).ReflectsIsomorphisms :=
  (fullyFaithfulSheafToPresheaf J A).reflectsIsomorphisms


/-- This is stated as a lemma to prevent class search from forming a loop since a sheaf morphism is
monic if and only if it is monic as a presheaf morphism (under suitable assumption). -/
theorem Sheaf.Hom.mono_of_presheaf_mono {F G : Sheaf J A} (f : F ⟶ G) [h : Mono f.1] : Mono f :=
  (sheafToPresheaf J A).mono_of_mono_map h


instance Sheaf.Hom.epi_of_presheaf_epi {F G : Sheaf J A} (f : F ⟶ G) [h : Epi f.1] : Epi f :=
  (sheafToPresheaf J A).epi_of_epi_map h


theorem isSheaf_iff_isSheaf_of_type (P : Cᵒᵖ ⥤ Type w) :
    Presheaf.IsSheaf J P ↔ Presieve.IsSheaf J P := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    ⊢ Iff (CategoryTheory.Presheaf.IsSheaf J P) (CategoryTheory.Presieve.IsSheaf J …
  -/
  constructor
    /-
      case mp
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      ⊢ CategoryTheory.Presheaf.IsSheaf J P → CategoryTheory.Presieve.IsSheaf J P
    -/
  · intro hP
    /-
      case mp
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      hP : CategoryTheory.Presheaf.IsSheaf J P
      ⊢ CategoryTheory.Presieve.IsSheaf J P
    -/
    refine Presieve.isSheaf_iso J ?_ (hP PUnit)
    /-
      case mp
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      hP : CategoryTheory.Presheaf.IsSheaf J P
      ⊢ CategoryTheory.Iso (P.comp (CategoryTheory.coyoneda.obj { unop := PUnit.{w + …
    -/
    exact isoWhiskerLeft _ Coyoneda.punitIso ≪≫ P.rightUnitor
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      ⊢ CategoryTheory.Presieve.IsSheaf J P → CategoryTheory.Presheaf.IsSheaf J P
    -/
  · intro hP X Y S hS z hz
    /-
      case mpr
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      hP : CategoryTheory.Presieve.IsSheaf J P
      X : Type w
      Y : C
      S : CategoryTheory.Sieve Y
      hS : Membership.mem (J Y) S
      z : CategoryTheory.Presieve.FamilyOfElements (P.comp (CategoryTheory.coyoneda. …
      hz : z.Compatible
      ⊢ ExistsUnique fun t => z.IsAmalgamation t
    -/
    refine ⟨fun x => (hP S hS).amalgamate (fun Z f hf => z f hf x) ?_, ?_, ?_⟩
      /-
        case mpr.refine_1
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        J : CategoryTheory.GrothendieckTopology C
        P : CategoryTheory.Functor (Opposite C) (Type w)
        hP : CategoryTheory.Presieve.IsSheaf J P
        X : Type w
        Y : C
        S : CategoryTheory.Sieve Y
        hS : Membership.mem (J Y) S
        z : CategoryTheory.Presieve.FamilyOfElements (P.comp (CategoryTheory.coyoneda. …
        hz : z.Compatible
        x : Opposite.unop { unop := X }
        ⊢ CategoryTheory.Presieve.FamilyOfElements.Compatible fun Z f hf => z f hf x
      -/
    · intro Y₁ Y₂ Z g₁ g₂ f₁ f₂ hf₁ hf₂ h
      /-
        case mpr.refine_1
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        J : CategoryTheory.GrothendieckTopology C
        P : CategoryTheory.Functor (Opposite C) (Type w)
        hP : CategoryTheory.Presieve.IsSheaf J P
        X : Type w
        Y : C
        S : CategoryTheory.Sieve Y
        hS : Membership.mem (J Y) S
        z : CategoryTheory.Presieve.FamilyOfElements (P.comp (CategoryTheory.coyoneda. …
        hz : z.Compatible
        x : Opposite.unop { unop := X }
        Y₁ Y₂ Z : C
        g₁ : Quiver.Hom Z Y₁
        g₂ : Quiver.Hom Z Y₂
        f₁ : Quiver.Hom Y₁ Y
        f₂ : Quiver.Hom Y₂ Y
        hf₁ : S.arrows f₁
        hf₂ : S.arrows f₂
        h : Eq (CategoryTheory.CategoryStruct.comp g₁ f₁) (CategoryTheory.CategoryStru …
        ⊢ Eq (P.map g₁.op ((fun Z f hf => z f hf x) Y₁ f₁ hf₁)) (P.map g₂.op ((fun Z f …
      -/
      exact congr_fun (hz g₁ g₂ hf₁ hf₂ h) x
      /-
        🎉 no goals
      -/
      /-
        case mpr.refine_2
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        J : CategoryTheory.GrothendieckTopology C
        P : CategoryTheory.Functor (Opposite C) (Type w)
        hP : CategoryTheory.Presieve.IsSheaf J P
        X : Type w
        Y : C
        S : CategoryTheory.Sieve Y
        hS : Membership.mem (J Y) S
        z : CategoryTheory.Presieve.FamilyOfElements (P.comp (CategoryTheory.coyoneda. …
        hz : z.Compatible
        ⊢ (fun t => z.IsAmalgamation t) fun x => ⋯.amalgamate (fun Z f hf => z f hf x) ⋯
      -/
    · intro Z f hf
      /-
        case mpr.refine_2
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        J : CategoryTheory.GrothendieckTopology C
        P : CategoryTheory.Functor (Opposite C) (Type w)
        hP : CategoryTheory.Presieve.IsSheaf J P
        X : Type w
        Y : C
        S : CategoryTheory.Sieve Y
        hS : Membership.mem (J Y) S
        z : CategoryTheory.Presieve.FamilyOfElements (P.comp (CategoryTheory.coyoneda. …
        hz : z.Compatible
        Z : C
        f : Quiver.Hom Z Y
        hf : S.arrows f
        ⊢ Eq ((P.comp (CategoryTheory.coyoneda.obj { unop := X })).map f.op fun x => ⋯ …
      -/
      funext x
      /-
        case mpr.refine_2.h
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        J : CategoryTheory.GrothendieckTopology C
        P : CategoryTheory.Functor (Opposite C) (Type w)
        hP : CategoryTheory.Presieve.IsSheaf J P
        X : Type w
        Y : C
        S : CategoryTheory.Sieve Y
        hS : Membership.mem (J Y) S
        z : CategoryTheory.Presieve.FamilyOfElements (P.comp (CategoryTheory.coyoneda. …
        hz : z.Compatible
        Z : C
        f : Quiver.Hom Z Y
        hf : S.arrows f
        x : Opposite.unop { unop := X }
        ⊢ Eq ((P.comp (CategoryTheory.coyoneda.obj { unop := X })).map f.op (fun x =>  …
      -/
      apply Presieve.IsSheafFor.valid_glue
      /-
        🎉 no goals
      -/
      /-
        case mpr.refine_3
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        J : CategoryTheory.GrothendieckTopology C
        P : CategoryTheory.Functor (Opposite C) (Type w)
        hP : CategoryTheory.Presieve.IsSheaf J P
        X : Type w
        Y : C
        S : CategoryTheory.Sieve Y
        hS : Membership.mem (J Y) S
        z : CategoryTheory.Presieve.FamilyOfElements (P.comp (CategoryTheory.coyoneda. …
        hz : z.Compatible
        ⊢ ∀ (y : (P.comp (CategoryTheory.coyoneda.obj { unop := X })).obj { unop := Y  …
      -/
    · intro y hy
      /-
        case mpr.refine_3
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        J : CategoryTheory.GrothendieckTopology C
        P : CategoryTheory.Functor (Opposite C) (Type w)
        hP : CategoryTheory.Presieve.IsSheaf J P
        X : Type w
        Y : C
        S : CategoryTheory.Sieve Y
        hS : Membership.mem (J Y) S
        z : CategoryTheory.Presieve.FamilyOfElements (P.comp (CategoryTheory.coyoneda. …
        hz : z.Compatible
        y : (P.comp (CategoryTheory.coyoneda.obj { unop := X })).obj { unop := Y }
        hy : z.IsAmalgamation y
        ⊢ Eq y fun x => ⋯.amalgamate (fun Z f hf => z f hf x) ⋯
      -/
      funext x
      /-
        case mpr.refine_3.h
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        J : CategoryTheory.GrothendieckTopology C
        P : CategoryTheory.Functor (Opposite C) (Type w)
        hP : CategoryTheory.Presieve.IsSheaf J P
        X : Type w
        Y : C
        S : CategoryTheory.Sieve Y
        hS : Membership.mem (J Y) S
        z : CategoryTheory.Presieve.FamilyOfElements (P.comp (CategoryTheory.coyoneda. …
        hz : z.Compatible
        y : (P.comp (CategoryTheory.coyoneda.obj { unop := X })).obj { unop := Y }
        hy : z.IsAmalgamation y
        x : Opposite.unop { unop := X }
        ⊢ Eq (y x) (⋯.amalgamate (fun Z f hf => z f hf x) ⋯)
      -/
      apply (hP S hS).isSeparatedFor.ext
      /-
        case mpr.refine_3.h
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        J : CategoryTheory.GrothendieckTopology C
        P : CategoryTheory.Functor (Opposite C) (Type w)
        hP : CategoryTheory.Presieve.IsSheaf J P
        X : Type w
        Y : C
        S : CategoryTheory.Sieve Y
        hS : Membership.mem (J Y) S
        z : CategoryTheory.Presieve.FamilyOfElements (P.comp (CategoryTheory.coyoneda. …
        hz : z.Compatible
        y : (P.comp (CategoryTheory.coyoneda.obj { unop := X })).obj { unop := Y }
        hy : z.IsAmalgamation y
        x : Opposite.unop { unop := X }
        ⊢ ∀ ⦃Y_1 : C⦄ ⦃f : Quiver.Hom Y_1 Y⦄, S.arrows f → Eq (P.map f.op (y x)) (P.ma …
      -/
      intro Y' f hf
      /-
        case mpr.refine_3.h
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        J : CategoryTheory.GrothendieckTopology C
        P : CategoryTheory.Functor (Opposite C) (Type w)
        hP : CategoryTheory.Presieve.IsSheaf J P
        X : Type w
        Y : C
        S : CategoryTheory.Sieve Y
        hS : Membership.mem (J Y) S
        z : CategoryTheory.Presieve.FamilyOfElements (P.comp (CategoryTheory.coyoneda. …
        hz : z.Compatible
        y : (P.comp (CategoryTheory.coyoneda.obj { unop := X })).obj { unop := Y }
        hy : z.IsAmalgamation y
        x : Opposite.unop { unop := X }
        Y' : C
        f : Quiver.Hom Y' Y
        hf : S.arrows f
        ⊢ Eq (P.map f.op (y x)) (P.map f.op (⋯.amalgamate (fun Z f hf => z f hf x) ⋯))
      -/
      rw [Presieve.IsSheafFor.valid_glue _ _ _ hf, ← hy _ hf]
      /-
        case mpr.refine_3.h
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        J : CategoryTheory.GrothendieckTopology C
        P : CategoryTheory.Functor (Opposite C) (Type w)
        hP : CategoryTheory.Presieve.IsSheaf J P
        X : Type w
        Y : C
        S : CategoryTheory.Sieve Y
        hS : Membership.mem (J Y) S
        z : CategoryTheory.Presieve.FamilyOfElements (P.comp (CategoryTheory.coyoneda. …
        hz : z.Compatible
        y : (P.comp (CategoryTheory.coyoneda.obj { unop := X })).obj { unop := Y }
        hy : z.IsAmalgamation y
        x : Opposite.unop { unop := X }
        Y' : C
        f : Quiver.Hom Y' Y
        hf : S.arrows f
        ⊢ Eq (P.map f.op (y x)) ((P.comp (CategoryTheory.coyoneda.obj { unop := X })). …
      -/
      rfl
      /-
        🎉 no goals
      -/


/-- The sheaf of sections guaranteed by the sheaf condition. -/
@[simps]
def sheafOver {A : Type u₂} [Category.{v₂} A] {J : GrothendieckTopology C} (ℱ : Sheaf J A) (E : A) :
    Sheaf J (Type _) where
  val := ℱ.val ⋙ coyoneda.obj (op E)
  cond := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      J✝ : CategoryTheory.GrothendieckTopology C
      A✝ : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} A✝
      A : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} A
      J : CategoryTheory.GrothendieckTopology C
      ℱ : CategoryTheory.Sheaf J A
      E : A
      ⊢ CategoryTheory.Presheaf.IsSheaf J (ℱ.val.comp (CategoryTheory.coyoneda.obj { …
    -/
    rw [isSheaf_iff_isSheaf_of_type]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      J✝ : CategoryTheory.GrothendieckTopology C
      A✝ : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} A✝
      A : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} A
      J : CategoryTheory.GrothendieckTopology C
      ℱ : CategoryTheory.Sheaf J A
      E : A
      ⊢ CategoryTheory.Presieve.IsSheaf J (ℱ.val.comp (CategoryTheory.coyoneda.obj { …
    -/
    exact ℱ.cond E
    /-
      🎉 no goals
    -/


variable {J} in
lemma Presheaf.IsSheaf.isSheafFor {P : Cᵒᵖ ⥤ Type w} (hP : Presheaf.IsSheaf J P)
    {X : C} (S : Sieve X) (hS : S ∈ J X) : Presieve.IsSheafFor P S.arrows := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    hP : CategoryTheory.Presheaf.IsSheaf J P
    X : C
    S : CategoryTheory.Sieve X
    hS : Membership.mem (J X) S
    ⊢ CategoryTheory.Presieve.IsSheafFor P S.arrows
  -/
  rw [isSheaf_iff_isSheaf_of_type] at hP
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    hP : CategoryTheory.Presieve.IsSheaf J P
    X : C
    S : CategoryTheory.Sieve X
    hS : Membership.mem (J X) S
    ⊢ CategoryTheory.Presieve.IsSheafFor P S.arrows
  -/
  exact hP S hS
  /-
    🎉 no goals
  -/


variable {A} in
lemma Presheaf.isSheaf_bot (P : Cᵒᵖ ⥤ A) : IsSheaf ⊥ P := fun _ ↦ Presieve.isSheaf_bot


/--
The category of sheaves on the bottom (trivial) Grothendieck topology is
equivalent to the category of presheaves.
-/
@[simps]
def sheafBotEquivalence : Sheaf (⊥ : GrothendieckTopology C) A ≌ Cᵒᵖ ⥤ A where
  functor := sheafToPresheaf _ _
  inverse :=
    { obj := fun P => ⟨P, Presheaf.isSheaf_bot P⟩
      map := fun f => ⟨f⟩ }
  unitIso := Iso.refl _
  counitIso := Iso.refl _


instance : Inhabited (Sheaf (⊥ : GrothendieckTopology C) (Type w)) :=
  ⟨(sheafBotEquivalence _).inverse.obj ((Functor.const _).obj default)⟩


/-- If the empty sieve is a cover of `X`, then `F(X)` is terminal. -/
def Sheaf.isTerminalOfBotCover (F : Sheaf J A) (X : C) (H : ⊥ ∈ J X) :
    IsTerminal (F.1.obj (op X)) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} A
    F : CategoryTheory.Sheaf J A
    X : C
    H : Membership.mem (J X) Bot.bot
    ⊢ CategoryTheory.Limits.IsTerminal (F.val.obj { unop := X })
  -/
  refine @IsTerminal.ofUnique _ _ _ ?_
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} A
    F : CategoryTheory.Sheaf J A
    X : C
    H : Membership.mem (J X) Bot.bot
    ⊢ (X_1 : A) → Unique (Quiver.Hom X_1 (F.val.obj { unop := X }))
  -/
  intro Y
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} A
    F : CategoryTheory.Sheaf J A
    X : C
    H : Membership.mem (J X) Bot.bot
    Y : A
    ⊢ Unique (Quiver.Hom Y (F.val.obj { unop := X }))
  -/
  choose t h using F.2 Y _ H (by tauto) (by tauto)
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} A
    F : CategoryTheory.Sheaf J A
    X : C
    H : Membership.mem (J X) Bot.bot
    Y : A
    t : (F.val.comp (CategoryTheory.coyoneda.obj { unop := Y })).obj { unop := X }
    h : And ((fun t => CategoryTheory.Presieve.FamilyOfElements.IsAmalgamation (fu …
    ⊢ Unique (Quiver.Hom Y (F.val.obj { unop := X }))
  -/
  exact ⟨⟨t⟩, fun a => h.2 a (by tauto)⟩
  /-
    🎉 no goals
  -/


instance sheafHomHasZSMul : SMul ℤ (P ⟶ Q) where
  smul n f :=
    Sheaf.Hom.mk
      { app := fun U => n • f.1.app U
        naturality := fun U V i => by
          /-
            C : Type u₁
            inst✝² : CategoryTheory.Category.{v₁, u₁} C
            J : CategoryTheory.GrothendieckTopology C
            A : Type u₂
            inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
            inst✝ : CategoryTheory.Preadditive A
            P Q : CategoryTheory.Sheaf J A
            n : Int
            f : Quiver.Hom P Q
            U V : Opposite C
            i : Quiver.Hom U V
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.val.map i) ((fun U => HSMul.hSMul  …
          -/
          induction' n using Int.induction_on with n ih n ih
            /-
              case hz
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              J : CategoryTheory.GrothendieckTopology C
              A : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
              inst✝ : CategoryTheory.Preadditive A
              P Q : CategoryTheory.Sheaf J A
              f : Quiver.Hom P Q
              U V : Opposite C
              i : Quiver.Hom U V
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.val.map i) ((fun U => HSMul.hSMul  …
            -/
          · simp only [zero_smul, comp_zero, zero_comp]
            /-
              🎉 no goals
            -/
          · simpa only [add_zsmul, one_zsmul, comp_add, NatTrans.naturality, add_comp,
              add_left_inj]
          · simpa only [sub_smul, one_zsmul, comp_sub, NatTrans.naturality, sub_comp,
              sub_left_inj] using ih }


instance : Sub (P ⟶ Q) where sub f g := Sheaf.Hom.mk <| f.1 - g.1


instance : Neg (P ⟶ Q) where neg f := Sheaf.Hom.mk <| -f.1


instance sheafHomHasNSMul : SMul ℕ (P ⟶ Q) where
  smul n f :=
    Sheaf.Hom.mk
      { app := fun U => n • f.1.app U
        naturality := fun U V i => by
          induction n with
          | zero => simp only [zero_smul, comp_zero, zero_comp]
          | succ n ih => simp only [Nat.succ_eq_add_one, add_smul, ih, one_nsmul, comp_add,
              NatTrans.naturality, add_comp] }


instance : Zero (P ⟶ Q) where zero := Sheaf.Hom.mk 0


instance : Add (P ⟶ Q) where add f g := Sheaf.Hom.mk <| f.1 + g.1


@[simp]
theorem Sheaf.Hom.add_app (f g : P ⟶ Q) (U) : (f + g).1.app U = f.1.app U + g.1.app U :=
  rfl


instance Sheaf.Hom.addCommGroup : AddCommGroup (P ⟶ Q) :=
  Function.Injective.addCommGroup (fun f : Sheaf.Hom P Q => f.1)
    (fun _ _ h => Sheaf.Hom.ext h) rfl (fun _ _ => rfl) (fun _ => rfl) (fun _ _ => rfl)
                   /-
                     C : Type u₁
                     inst✝² : CategoryTheory.Category.{v₁, u₁} C
                     J : CategoryTheory.GrothendieckTopology C
                     A : Type u₂
                     inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
                     inst✝ : CategoryTheory.Preadditive A
                     P Q : CategoryTheory.Sheaf J A
                     x✝¹ : Quiver.Hom P Q
                     x✝ : Nat
                     ⊢ Eq ((fun f => f.val) (HSMul.hSMul x✝ x✝¹)) (HSMul.hSMul x✝ ((fun f => f.val) …
                   -/
                   /-
                     🎉 no goals
                   -/
    (fun _ _ => by aesop_cat) (fun _ _ => by aesop_cat)
                                             /-
                                               🎉 no goals
                                             -/


instance : Preadditive (Sheaf J A) where
  homGroup _ _ := Sheaf.Hom.addCommGroup


/-- When `P` is a sheaf and `S` is a cover, the associated multifork is a limit. -/
def isLimitOfIsSheaf {X : C} (S : J.Cover X) (hP : IsSheaf J P) : IsLimit (S.multifork P) where
  lift := fun E : Multifork _ => hP.amalgamate S (fun _ => E.ι _)
    (fun _ _ r => E.condition ⟨_, _, r⟩)
  fac := by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} A
      A' : Type u₂
      inst✝¹ : CategoryTheory.Category.{max v₁ u₁, u₂} A'
      B : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} B
      J : CategoryTheory.GrothendieckTopology C
      U : C
      R : CategoryTheory.Presieve U
      P : CategoryTheory.Functor (Opposite C) A
      P' : CategoryTheory.Functor (Opposite C) A'
      X : C
      S : J.Cover X
      hP : CategoryTheory.Presheaf.IsSheaf J P
      ⊢ ∀ (s : CategoryTheory.Limits.Cone (S.index P).multicospan) (j : CategoryTheo …
    -/
    rintro (E : Multifork _) (a | b)
      /-
        case left
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        A : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} A
        A' : Type u₂
        inst✝¹ : CategoryTheory.Category.{max v₁ u₁, u₂} A'
        B : Type u₃
        inst✝ : CategoryTheory.Category.{v₃, u₃} B
        J : CategoryTheory.GrothendieckTopology C
        U : C
        R : CategoryTheory.Presieve U
        P : CategoryTheory.Functor (Opposite C) A
        P' : CategoryTheory.Functor (Opposite C) A'
        X : C
        S : J.Cover X
        hP : CategoryTheory.Presheaf.IsSheaf J P
        E : CategoryTheory.Limits.Multifork (S.index P)
        a : (S.index P).L
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun E => hP.amalgamate S (fun x =>  …
      -/
    · apply hP.amalgamate_map
      /-
        🎉 no goals
      -/
    · rw [← E.w (WalkingMulticospan.Hom.fst b),
        ← (S.multifork P).w (WalkingMulticospan.Hom.fst b), ← assoc]
      /-
        case right
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        A : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} A
        A' : Type u₂
        inst✝¹ : CategoryTheory.Category.{max v₁ u₁, u₂} A'
        B : Type u₃
        inst✝ : CategoryTheory.Category.{v₃, u₃} B
        J : CategoryTheory.GrothendieckTopology C
        U : C
        R : CategoryTheory.Presieve U
        P : CategoryTheory.Functor (Opposite C) A
        P' : CategoryTheory.Functor (Opposite C) A'
        X : C
        S : J.Cover X
        hP : CategoryTheory.Presheaf.IsSheaf J P
        E : CategoryTheory.Limits.Multifork (S.index P)
        b : (S.index P).R
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      congr 1
      /-
        case right.e_a
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        A : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} A
        A' : Type u₂
        inst✝¹ : CategoryTheory.Category.{max v₁ u₁, u₂} A'
        B : Type u₃
        inst✝ : CategoryTheory.Category.{v₃, u₃} B
        J : CategoryTheory.GrothendieckTopology C
        U : C
        R : CategoryTheory.Presieve U
        P : CategoryTheory.Functor (Opposite C) A
        P' : CategoryTheory.Functor (Opposite C) A'
        X : C
        S : J.Cover X
        hP : CategoryTheory.Presheaf.IsSheaf J P
        E : CategoryTheory.Limits.Multifork (S.index P)
        b : (S.index P).R
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun E => hP.amalgamate S (fun x =>  …
      -/
      apply hP.amalgamate_map
      /-
        🎉 no goals
      -/
  uniq := by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} A
      A' : Type u₂
      inst✝¹ : CategoryTheory.Category.{max v₁ u₁, u₂} A'
      B : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} B
      J : CategoryTheory.GrothendieckTopology C
      U : C
      R : CategoryTheory.Presieve U
      P : CategoryTheory.Functor (Opposite C) A
      P' : CategoryTheory.Functor (Opposite C) A'
      X : C
      S : J.Cover X
      hP : CategoryTheory.Presheaf.IsSheaf J P
      ⊢ ∀ (s : CategoryTheory.Limits.Cone (S.index P).multicospan) (m : Quiver.Hom s …
    -/
    rintro (E : Multifork _) m hm
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} A
      A' : Type u₂
      inst✝¹ : CategoryTheory.Category.{max v₁ u₁, u₂} A'
      B : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} B
      J : CategoryTheory.GrothendieckTopology C
      U : C
      R : CategoryTheory.Presieve U
      P : CategoryTheory.Functor (Opposite C) A
      P' : CategoryTheory.Functor (Opposite C) A'
      X : C
      S : J.Cover X
      hP : CategoryTheory.Presheaf.IsSheaf J P
      E : CategoryTheory.Limits.Multifork (S.index P)
      m : Quiver.Hom E.pt (S.multifork P).pt
      hm : ∀ (j : CategoryTheory.Limits.WalkingMulticospan (S.index P).fstTo (S.inde …
      ⊢ Eq m ((fun E => hP.amalgamate S (fun x => E.ι x) ⋯) E)
    -/
    apply hP.hom_ext S
    /-
      case h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} A
      A' : Type u₂
      inst✝¹ : CategoryTheory.Category.{max v₁ u₁, u₂} A'
      B : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} B
      J : CategoryTheory.GrothendieckTopology C
      U : C
      R : CategoryTheory.Presieve U
      P : CategoryTheory.Functor (Opposite C) A
      P' : CategoryTheory.Functor (Opposite C) A'
      X : C
      S : J.Cover X
      hP : CategoryTheory.Presheaf.IsSheaf J P
      E : CategoryTheory.Limits.Multifork (S.index P)
      m : Quiver.Hom E.pt (S.multifork P).pt
      hm : ∀ (j : CategoryTheory.Limits.WalkingMulticospan (S.index P).fstTo (S.inde …
      ⊢ ∀ (I : S.Arrow), Eq (CategoryTheory.CategoryStruct.comp m (P.map I.f.op)) (C …
    -/
    intro I
    /-
      case h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} A
      A' : Type u₂
      inst✝¹ : CategoryTheory.Category.{max v₁ u₁, u₂} A'
      B : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} B
      J : CategoryTheory.GrothendieckTopology C
      U : C
      R : CategoryTheory.Presieve U
      P : CategoryTheory.Functor (Opposite C) A
      P' : CategoryTheory.Functor (Opposite C) A'
      X : C
      S : J.Cover X
      hP : CategoryTheory.Presheaf.IsSheaf J P
      E : CategoryTheory.Limits.Multifork (S.index P)
      m : Quiver.Hom E.pt (S.multifork P).pt
      hm : ∀ (j : CategoryTheory.Limits.WalkingMulticospan (S.index P).fstTo (S.inde …
      I : S.Arrow
      ⊢ Eq (CategoryTheory.CategoryStruct.comp m (P.map I.f.op)) (CategoryTheory.Cat …
    -/
    erw [hm (WalkingMulticospan.left I)]
    /-
      case h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} A
      A' : Type u₂
      inst✝¹ : CategoryTheory.Category.{max v₁ u₁, u₂} A'
      B : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} B
      J : CategoryTheory.GrothendieckTopology C
      U : C
      R : CategoryTheory.Presieve U
      P : CategoryTheory.Functor (Opposite C) A
      P' : CategoryTheory.Functor (Opposite C) A'
      X : C
      S : J.Cover X
      hP : CategoryTheory.Presheaf.IsSheaf J P
      E : CategoryTheory.Limits.Multifork (S.index P)
      m : Quiver.Hom E.pt (S.multifork P).pt
      hm : ∀ (j : CategoryTheory.Limits.WalkingMulticospan (S.index P).fstTo (S.inde …
      I : S.Arrow
      ⊢ Eq (E.π.app (CategoryTheory.Limits.WalkingMulticospan.left I)) (CategoryTheo …
    -/
    symm
    /-
      case h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} A
      A' : Type u₂
      inst✝¹ : CategoryTheory.Category.{max v₁ u₁, u₂} A'
      B : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} B
      J : CategoryTheory.GrothendieckTopology C
      U : C
      R : CategoryTheory.Presieve U
      P : CategoryTheory.Functor (Opposite C) A
      P' : CategoryTheory.Functor (Opposite C) A'
      X : C
      S : J.Cover X
      hP : CategoryTheory.Presheaf.IsSheaf J P
      E : CategoryTheory.Limits.Multifork (S.index P)
      m : Quiver.Hom E.pt (S.multifork P).pt
      hm : ∀ (j : CategoryTheory.Limits.WalkingMulticospan (S.index P).fstTo (S.inde …
      I : S.Arrow
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun E => hP.amalgamate S (fun x =>  …
    -/
    apply hP.amalgamate_map
    /-
      🎉 no goals
    -/


theorem isSheaf_iff_multifork :
    IsSheaf J P ↔ ∀ (X : C) (S : J.Cover X), Nonempty (IsLimit (S.multifork P)) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} A
    J : CategoryTheory.GrothendieckTopology C
    P : CategoryTheory.Functor (Opposite C) A
    ⊢ Iff (CategoryTheory.Presheaf.IsSheaf J P) (∀ (X : C) (S : J.Cover X), Nonemp …
  -/
  refine ⟨fun hP X S => ⟨isLimitOfIsSheaf _ _ _ hP⟩, ?_⟩
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} A
    J : CategoryTheory.GrothendieckTopology C
    P : CategoryTheory.Functor (Opposite C) A
    ⊢ (∀ (X : C) (S : J.Cover X), Nonempty (CategoryTheory.Limits.IsLimit (S.multi …
  -/
  intro h E X S hS x hx
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} A
    J : CategoryTheory.GrothendieckTopology C
    P : CategoryTheory.Functor (Opposite C) A
    h : ∀ (X : C) (S : J.Cover X), Nonempty (CategoryTheory.Limits.IsLimit (S.mult …
    E : A
    X : C
    S : CategoryTheory.Sieve X
    hS : Membership.mem (J X) S
    x : CategoryTheory.Presieve.FamilyOfElements (P.comp (CategoryTheory.coyoneda. …
    hx : x.Compatible
    ⊢ ExistsUnique fun t => x.IsAmalgamation t
  -/
  let T : J.Cover X := ⟨S, hS⟩
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} A
    J : CategoryTheory.GrothendieckTopology C
    P : CategoryTheory.Functor (Opposite C) A
    h : ∀ (X : C) (S : J.Cover X), Nonempty (CategoryTheory.Limits.IsLimit (S.mult …
    E : A
    X : C
    S : CategoryTheory.Sieve X
    hS : Membership.mem (J X) S
    x : CategoryTheory.Presieve.FamilyOfElements (P.comp (CategoryTheory.coyoneda. …
    hx : x.Compatible
    T : J.Cover X := ⟨S, hS⟩
    ⊢ ExistsUnique fun t => x.IsAmalgamation t
  -/
  obtain ⟨hh⟩ := h _ T
  let K : Multifork (T.index P) := Multifork.ofι _ E (fun I => x I.f I.hf)
    (fun I => hx _ _ _ _ I.r.w)
  /-
    case intro
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} A
    J : CategoryTheory.GrothendieckTopology C
    P : CategoryTheory.Functor (Opposite C) A
    h : ∀ (X : C) (S : J.Cover X), Nonempty (CategoryTheory.Limits.IsLimit (S.mult …
    E : A
    X : C
    S : CategoryTheory.Sieve X
    hS : Membership.mem (J X) S
    x : CategoryTheory.Presieve.FamilyOfElements (P.comp (CategoryTheory.coyoneda. …
    hx : x.Compatible
    T : J.Cover X := ⟨S, hS⟩
    hh : CategoryTheory.Limits.IsLimit (T.multifork P)
    K : CategoryTheory.Limits.Multifork (T.index P) := CategoryTheory.Limits.Multi …
    ⊢ ExistsUnique fun t => x.IsAmalgamation t
  -/
  use hh.lift K
  /-
    case h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} A
    J : CategoryTheory.GrothendieckTopology C
    P : CategoryTheory.Functor (Opposite C) A
    h : ∀ (X : C) (S : J.Cover X), Nonempty (CategoryTheory.Limits.IsLimit (S.mult …
    E : A
    X : C
    S : CategoryTheory.Sieve X
    hS : Membership.mem (J X) S
    x : CategoryTheory.Presieve.FamilyOfElements (P.comp (CategoryTheory.coyoneda. …
    hx : x.Compatible
    T : J.Cover X := ⟨S, hS⟩
    hh : CategoryTheory.Limits.IsLimit (T.multifork P)
    K : CategoryTheory.Limits.Multifork (T.index P) := CategoryTheory.Limits.Multi …
    ⊢ And ((fun t => x.IsAmalgamation t) (hh.lift K)) (∀ (y : (P.comp (CategoryThe …
  -/
  dsimp; constructor
    /-
      case h.left
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} A
      J : CategoryTheory.GrothendieckTopology C
      P : CategoryTheory.Functor (Opposite C) A
      h : ∀ (X : C) (S : J.Cover X), Nonempty (CategoryTheory.Limits.IsLimit (S.mult …
      E : A
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J X) S
      x : CategoryTheory.Presieve.FamilyOfElements (P.comp (CategoryTheory.coyoneda. …
      hx : x.Compatible
      T : J.Cover X := ⟨S, hS⟩
      hh : CategoryTheory.Limits.IsLimit (T.multifork P)
      K : CategoryTheory.Limits.Multifork (T.index P) := CategoryTheory.Limits.Multi …
      ⊢ x.IsAmalgamation (hh.lift K)
    -/
  · intro Y f hf
    /-
      case h.left
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} A
      J : CategoryTheory.GrothendieckTopology C
      P : CategoryTheory.Functor (Opposite C) A
      h : ∀ (X : C) (S : J.Cover X), Nonempty (CategoryTheory.Limits.IsLimit (S.mult …
      E : A
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J X) S
      x : CategoryTheory.Presieve.FamilyOfElements (P.comp (CategoryTheory.coyoneda. …
      hx : x.Compatible
      T : J.Cover X := ⟨S, hS⟩
      hh : CategoryTheory.Limits.IsLimit (T.multifork P)
      K : CategoryTheory.Limits.Multifork (T.index P) := CategoryTheory.Limits.Multi …
      Y : C
      f : Quiver.Hom Y X
      hf : S.arrows f
      ⊢ Eq ((P.comp (CategoryTheory.coyoneda.obj { unop := E })).map f.op (hh.lift K …
    -/
    apply hh.fac K (WalkingMulticospan.left ⟨Y, f, hf⟩)
    /-
      🎉 no goals
    -/
    /-
      case h.right
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} A
      J : CategoryTheory.GrothendieckTopology C
      P : CategoryTheory.Functor (Opposite C) A
      h : ∀ (X : C) (S : J.Cover X), Nonempty (CategoryTheory.Limits.IsLimit (S.mult …
      E : A
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J X) S
      x : CategoryTheory.Presieve.FamilyOfElements (P.comp (CategoryTheory.coyoneda. …
      hx : x.Compatible
      T : J.Cover X := ⟨S, hS⟩
      hh : CategoryTheory.Limits.IsLimit (T.multifork P)
      K : CategoryTheory.Limits.Multifork (T.index P) := CategoryTheory.Limits.Multi …
      ⊢ ∀ (y : Quiver.Hom E (P.obj { unop := X })), x.IsAmalgamation y → Eq y (hh.li …
    -/
  · intro e he
    /-
      case h.right
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} A
      J : CategoryTheory.GrothendieckTopology C
      P : CategoryTheory.Functor (Opposite C) A
      h : ∀ (X : C) (S : J.Cover X), Nonempty (CategoryTheory.Limits.IsLimit (S.mult …
      E : A
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J X) S
      x : CategoryTheory.Presieve.FamilyOfElements (P.comp (CategoryTheory.coyoneda. …
      hx : x.Compatible
      T : J.Cover X := ⟨S, hS⟩
      hh : CategoryTheory.Limits.IsLimit (T.multifork P)
      K : CategoryTheory.Limits.Multifork (T.index P) := CategoryTheory.Limits.Multi …
      e : Quiver.Hom E (P.obj { unop := X })
      he : x.IsAmalgamation e
      ⊢ Eq e (hh.lift K)
    -/
    apply hh.uniq K
    /-
      case h.right.x
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} A
      J : CategoryTheory.GrothendieckTopology C
      P : CategoryTheory.Functor (Opposite C) A
      h : ∀ (X : C) (S : J.Cover X), Nonempty (CategoryTheory.Limits.IsLimit (S.mult …
      E : A
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J X) S
      x : CategoryTheory.Presieve.FamilyOfElements (P.comp (CategoryTheory.coyoneda. …
      hx : x.Compatible
      T : J.Cover X := ⟨S, hS⟩
      hh : CategoryTheory.Limits.IsLimit (T.multifork P)
      K : CategoryTheory.Limits.Multifork (T.index P) := CategoryTheory.Limits.Multi …
      e : Quiver.Hom E (P.obj { unop := X })
      he : x.IsAmalgamation e
      ⊢ ∀ (j : CategoryTheory.Limits.WalkingMulticospan (T.index P).fstTo (T.index P …
    -/
    rintro (a | b)
      /-
        case h.right.x.left
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        A : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} A
        J : CategoryTheory.GrothendieckTopology C
        P : CategoryTheory.Functor (Opposite C) A
        h : ∀ (X : C) (S : J.Cover X), Nonempty (CategoryTheory.Limits.IsLimit (S.mult …
        E : A
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem (J X) S
        x : CategoryTheory.Presieve.FamilyOfElements (P.comp (CategoryTheory.coyoneda. …
        hx : x.Compatible
        T : J.Cover X := ⟨S, hS⟩
        hh : CategoryTheory.Limits.IsLimit (T.multifork P)
        K : CategoryTheory.Limits.Multifork (T.index P) := CategoryTheory.Limits.Multi …
        e : Quiver.Hom E (P.obj { unop := X })
        he : x.IsAmalgamation e
        a : (T.index P).L
        ⊢ Eq (CategoryTheory.CategoryStruct.comp e ((T.multifork P).π.app (CategoryThe …
      -/
    · apply he
      /-
        🎉 no goals
      -/
    · rw [← K.w (WalkingMulticospan.Hom.fst b), ←
        (T.multifork P).w (WalkingMulticospan.Hom.fst b), ← assoc]
      /-
        case h.right.x.right
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        A : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} A
        J : CategoryTheory.GrothendieckTopology C
        P : CategoryTheory.Functor (Opposite C) A
        h : ∀ (X : C) (S : J.Cover X), Nonempty (CategoryTheory.Limits.IsLimit (S.mult …
        E : A
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem (J X) S
        x : CategoryTheory.Presieve.FamilyOfElements (P.comp (CategoryTheory.coyoneda. …
        hx : x.Compatible
        T : J.Cover X := ⟨S, hS⟩
        hh : CategoryTheory.Limits.IsLimit (T.multifork P)
        K : CategoryTheory.Limits.Multifork (T.index P) := CategoryTheory.Limits.Multi …
        e : Quiver.Hom E (P.obj { unop := X })
        he : x.IsAmalgamation e
        b : (T.index P).R
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp e …
      -/
      congr 1
      /-
        case h.right.x.right.e_a
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        A : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} A
        J : CategoryTheory.GrothendieckTopology C
        P : CategoryTheory.Functor (Opposite C) A
        h : ∀ (X : C) (S : J.Cover X), Nonempty (CategoryTheory.Limits.IsLimit (S.mult …
        E : A
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem (J X) S
        x : CategoryTheory.Presieve.FamilyOfElements (P.comp (CategoryTheory.coyoneda. …
        hx : x.Compatible
        T : J.Cover X := ⟨S, hS⟩
        hh : CategoryTheory.Limits.IsLimit (T.multifork P)
        K : CategoryTheory.Limits.Multifork (T.index P) := CategoryTheory.Limits.Multi …
        e : Quiver.Hom E (P.obj { unop := X })
        he : x.IsAmalgamation e
        b : (T.index P).R
        ⊢ Eq (CategoryTheory.CategoryStruct.comp e ((T.multifork P).π.app (CategoryThe …
      -/
      apply he
      /-
        🎉 no goals
      -/


variable {J P} in
/-- If `F : Cᵒᵖ ⥤ A` is a sheaf for a Grothendieck topology `J` on `C`,
and `S` is a cover of `X : C`, then the multifork `S.multifork F` is limit. -/
def IsSheaf.isLimitMultifork
    (hP : Presheaf.IsSheaf J P) {X : C} (S : J.Cover X) : IsLimit (S.multifork P) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} A
    A' : Type u₂
    inst✝¹ : CategoryTheory.Category.{max v₁ u₁, u₂} A'
    B : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} B
    J : CategoryTheory.GrothendieckTopology C
    U : C
    R : CategoryTheory.Presieve U
    P : CategoryTheory.Functor (Opposite C) A
    P' : CategoryTheory.Functor (Opposite C) A'
    hP : CategoryTheory.Presheaf.IsSheaf J P
    X : C
    S : J.Cover X
    ⊢ CategoryTheory.Limits.IsLimit (S.multifork P)
  -/
  rw [Presheaf.isSheaf_iff_multifork] at hP
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} A
    A' : Type u₂
    inst✝¹ : CategoryTheory.Category.{max v₁ u₁, u₂} A'
    B : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} B
    J : CategoryTheory.GrothendieckTopology C
    U : C
    R : CategoryTheory.Presieve U
    P : CategoryTheory.Functor (Opposite C) A
    P' : CategoryTheory.Functor (Opposite C) A'
    hP : ∀ (X : C) (S : J.Cover X), Nonempty (CategoryTheory.Limits.IsLimit (S.mul …
    X : C
    S : J.Cover X
    ⊢ CategoryTheory.Limits.IsLimit (S.multifork P)
  -/
  exact (hP X S).some
  /-
    🎉 no goals
  -/


theorem isSheaf_iff_multiequalizer [∀ (X : C) (S : J.Cover X), HasMultiequalizer (S.index P)] :
    IsSheaf J P ↔ ∀ (X : C) (S : J.Cover X), IsIso (S.toMultiequalizer P) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
    J : CategoryTheory.GrothendieckTopology C
    P : CategoryTheory.Functor (Opposite C) A
    inst✝ : ∀ (X : C) (S : J.Cover X), CategoryTheory.Limits.HasMultiequalizer (S. …
    ⊢ Iff (CategoryTheory.Presheaf.IsSheaf J P) (∀ (X : C) (S : J.Cover X), Catego …
  -/
  rw [isSheaf_iff_multifork]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
    J : CategoryTheory.GrothendieckTopology C
    P : CategoryTheory.Functor (Opposite C) A
    inst✝ : ∀ (X : C) (S : J.Cover X), CategoryTheory.Limits.HasMultiequalizer (S. …
    ⊢ Iff (∀ (X : C) (S : J.Cover X), Nonempty (CategoryTheory.Limits.IsLimit (S.m …
  -/
  refine forall₂_congr fun X S => ⟨?_, ?_⟩
    /-
      case refine_1
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
      J : CategoryTheory.GrothendieckTopology C
      P : CategoryTheory.Functor (Opposite C) A
      inst✝ : ∀ (X : C) (S : J.Cover X), CategoryTheory.Limits.HasMultiequalizer (S. …
      X : C
      S : J.Cover X
      ⊢ Nonempty (CategoryTheory.Limits.IsLimit (S.multifork P)) → CategoryTheory.Is …
    -/
  · rintro ⟨h⟩
    let e : P.obj (op X) ≅ multiequalizer (S.index P) :=
      h.conePointUniqueUpToIso (limit.isLimit _)
    /-
      case refine_1.intro
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
      J : CategoryTheory.GrothendieckTopology C
      P : CategoryTheory.Functor (Opposite C) A
      inst✝ : ∀ (X : C) (S : J.Cover X), CategoryTheory.Limits.HasMultiequalizer (S. …
      X : C
      S : J.Cover X
      h : CategoryTheory.Limits.IsLimit (S.multifork P)
      e : CategoryTheory.Iso (P.obj { unop := X }) (CategoryTheory.Limits.multiequal …
      ⊢ CategoryTheory.IsIso (S.toMultiequalizer P)
    -/
    exact (inferInstance : IsIso e.hom)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
      J : CategoryTheory.GrothendieckTopology C
      P : CategoryTheory.Functor (Opposite C) A
      inst✝ : ∀ (X : C) (S : J.Cover X), CategoryTheory.Limits.HasMultiequalizer (S. …
      X : C
      S : J.Cover X
      ⊢ CategoryTheory.IsIso (S.toMultiequalizer P) → Nonempty (CategoryTheory.Limit …
    -/
  · intro h
    /-
      case refine_2
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
      J : CategoryTheory.GrothendieckTopology C
      P : CategoryTheory.Functor (Opposite C) A
      inst✝ : ∀ (X : C) (S : J.Cover X), CategoryTheory.Limits.HasMultiequalizer (S. …
      X : C
      S : J.Cover X
      h : CategoryTheory.IsIso (S.toMultiequalizer P)
      ⊢ Nonempty (CategoryTheory.Limits.IsLimit (S.multifork P))
    -/
    refine ⟨IsLimit.ofIsoLimit (limit.isLimit _) (Cones.ext ?_ ?_)⟩
      /-
        case refine_2.refine_1
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        A : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
        J : CategoryTheory.GrothendieckTopology C
        P : CategoryTheory.Functor (Opposite C) A
        inst✝ : ∀ (X : C) (S : J.Cover X), CategoryTheory.Limits.HasMultiequalizer (S. …
        X : C
        S : J.Cover X
        h : CategoryTheory.IsIso (S.toMultiequalizer P)
        ⊢ CategoryTheory.Iso (CategoryTheory.Limits.limit.cone (S.index P).multicospan …
      -/
    · apply (@asIso _ _ _ _ _ h).symm
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        A : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
        J : CategoryTheory.GrothendieckTopology C
        P : CategoryTheory.Functor (Opposite C) A
        inst✝ : ∀ (X : C) (S : J.Cover X), CategoryTheory.Limits.HasMultiequalizer (S. …
        X : C
        S : J.Cover X
        h : CategoryTheory.IsIso (S.toMultiequalizer P)
        ⊢ ∀ (j : CategoryTheory.Limits.WalkingMulticospan (S.index P).fstTo (S.index P …
      -/
    · intro a
      /-
        case refine_2.refine_2
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        A : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
        J : CategoryTheory.GrothendieckTopology C
        P : CategoryTheory.Functor (Opposite C) A
        inst✝ : ∀ (X : C) (S : J.Cover X), CategoryTheory.Limits.HasMultiequalizer (S. …
        X : C
        S : J.Cover X
        h : CategoryTheory.IsIso (S.toMultiequalizer P)
        a : CategoryTheory.Limits.WalkingMulticospan (S.index P).fstTo (S.index P).sndTo
        ⊢ Eq ((CategoryTheory.Limits.limit.cone (S.index P).multicospan).π.app a) (Cat …
      -/
      symm
      /-
        case refine_2.refine_2
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        A : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
        J : CategoryTheory.GrothendieckTopology C
        P : CategoryTheory.Functor (Opposite C) A
        inst✝ : ∀ (X : C) (S : J.Cover X), CategoryTheory.Limits.HasMultiequalizer (S. …
        X : C
        S : J.Cover X
        h : CategoryTheory.IsIso (S.toMultiequalizer P)
        a : CategoryTheory.Limits.WalkingMulticospan (S.index P).fstTo (S.index P).sndTo
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.asIso (S.toMultiequal …
      -/
      erw [IsIso.inv_comp_eq]
      /-
        case refine_2.refine_2
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        A : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
        J : CategoryTheory.GrothendieckTopology C
        P : CategoryTheory.Functor (Opposite C) A
        inst✝ : ∀ (X : C) (S : J.Cover X), CategoryTheory.Limits.HasMultiequalizer (S. …
        X : C
        S : J.Cover X
        h : CategoryTheory.IsIso (S.toMultiequalizer P)
        a : CategoryTheory.Limits.WalkingMulticospan (S.index P).fstTo (S.index P).sndTo
        ⊢ Eq ((S.multifork P).π.app a) (CategoryTheory.CategoryStruct.comp (S.toMultie …
      -/
      simp
      /-
        🎉 no goals
      -/


/--
The middle object of the fork diagram given in Equation (3) of [MM92], as well as the fork diagram
of <https://stacks.math.columbia.edu/tag/00VM>.
-/
def firstObj : A :=
  ∏ᶜ fun f : ΣV, { f : V ⟶ U // R f } => P.obj (op f.1)


/--
The left morphism of the fork diagram given in Equation (3) of [MM92], as well as the fork diagram
of <https://stacks.math.columbia.edu/tag/00VM>.
-/
def forkMap : P.obj (op U) ⟶ firstObj R P :=
  Pi.lift fun f => P.map f.2.1.op


/-- The rightmost object of the fork diagram of https://stacks.math.columbia.edu/tag/00VM, which
contains the data used to check a family of elements for a presieve is compatible.
-/
def secondObj : A :=
  ∏ᶜ fun fg : (ΣV, { f : V ⟶ U // R f }) × ΣW, { g : W ⟶ U // R g } =>
    P.obj (op (pullback fg.1.2.1 fg.2.2.1))


/-- The map `pr₀*` of <https://stacks.math.columbia.edu/tag/00VM>. -/
def firstMap : firstObj R P ⟶ secondObj R P :=
  Pi.lift fun _ => Pi.π _ _ ≫ P.map (pullback.fst _ _).op


/-- The map `pr₁*` of <https://stacks.math.columbia.edu/tag/00VM>. -/
def secondMap : firstObj R P ⟶ secondObj R P :=
  Pi.lift fun _ => Pi.π _ _ ≫ P.map (pullback.snd _ _).op


theorem w : forkMap R P ≫ firstMap R P = forkMap R P ≫ secondMap R P := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} A
    U : C
    R : CategoryTheory.Presieve U
    P : CategoryTheory.Functor (Opposite C) A
    inst✝¹ : CategoryTheory.Limits.HasProducts A
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Presheaf.forkMap R P) …
  -/
  apply limit.hom_ext
  /-
    case w
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} A
    U : C
    R : CategoryTheory.Presieve U
    P : CategoryTheory.Functor (Opposite C) A
    inst✝¹ : CategoryTheory.Limits.HasProducts A
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    ⊢ ∀ (j : CategoryTheory.Discrete (Prod (Sigma fun V => Subtype fun f => R f) ( …
  -/
  rintro ⟨⟨Y, f, hf⟩, ⟨Z, g, hg⟩⟩
  simp only [firstMap, secondMap, forkMap, limit.lift_π, limit.lift_π_assoc, assoc, Fan.mk_π_app,
    Subtype.coe_mk]
  /-
    case w.mk.mk.mk.mk.mk.mk
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} A
    U : C
    R : CategoryTheory.Presieve U
    P : CategoryTheory.Functor (Opposite C) A
    inst✝¹ : CategoryTheory.Limits.HasProducts A
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    Y : C
    f : Quiver.Hom Y U
    hf : R f
    Z : C
    g : Quiver.Hom Z U
    hg : R g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.map f.op) (P.map (CategoryTheory.L …
  -/
  rw [← P.map_comp, ← op_comp, pullback.condition]
  /-
    case w.mk.mk.mk.mk.mk.mk
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} A
    U : C
    R : CategoryTheory.Presieve U
    P : CategoryTheory.Functor (Opposite C) A
    inst✝¹ : CategoryTheory.Limits.HasProducts A
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    Y : C
    f : Quiver.Hom Y U
    hf : R f
    Z : C
    g : Quiver.Hom Z U
    hg : R g
    ⊢ Eq (P.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbac …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- An alternative definition of the sheaf condition in terms of equalizers. This is shown to be
equivalent in `CategoryTheory.Presheaf.isSheaf_iff_isSheaf'`.
-/
def IsSheaf' (P : Cᵒᵖ ⥤ A) : Prop :=
  ∀ (U : C) (R : Presieve U) (_ : generate R ∈ J U), Nonempty (IsLimit (Fork.ofι _ (w R P)))

-- Again I wonder whether `UnivLE` can somehow be used to allow `s` to take
-- values in a more general universe.

/-- (Implementation). An auxiliary lemma to convert between sheaf conditions. -/
def isSheafForIsSheafFor' (P : Cᵒᵖ ⥤ A) (s : A ⥤ Type max v₁ u₁)
    [∀ J, PreservesLimitsOfShape (Discrete.{max v₁ u₁} J) s] (U : C) (R : Presieve U) :
    IsLimit (s.mapCone (Fork.ofι _ (w R P))) ≃
      IsLimit (Fork.ofι _ (Equalizer.Presieve.w (P ⋙ s) R)) := by
  let e : parallelPair (s.map (firstMap R P)) (s.map (secondMap R P)) ≅
    parallelPair (Equalizer.Presieve.firstMap (P ⋙ s) R)
      (Equalizer.Presieve.secondMap (P ⋙ s) R) := by
    refine parallelPair.ext (PreservesProduct.iso s _) ((PreservesProduct.iso s _))
      (limit.hom_ext (fun j => ?_)) (limit.hom_ext (fun j => ?_))
    · dsimp [Equalizer.Presieve.firstMap, firstMap]
      simp only [map_lift_piComparison, Functor.map_comp, limit.lift_π, Fan.mk_pt,
        Fan.mk_π_app, assoc, piComparison_comp_π_assoc]
    · dsimp [Equalizer.Presieve.secondMap, secondMap]
      simp only [map_lift_piComparison, Functor.map_comp, limit.lift_π, Fan.mk_pt,
        Fan.mk_π_app, assoc, piComparison_comp_π_assoc]
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₂, u₂} A
    A' : Type u₂
    inst✝⁵ : CategoryTheory.Category.{max v₁ u₁, u₂} A'
    B : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₃, u₃} B
    J : CategoryTheory.GrothendieckTopology C
    U✝ : C
    R✝ : CategoryTheory.Presieve U✝
    P✝ : CategoryTheory.Functor (Opposite C) A
    P' : CategoryTheory.Functor (Opposite C) A'
    inst✝³ : CategoryTheory.Limits.HasProducts A
    inst✝² : CategoryTheory.Limits.HasProducts A'
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    P : CategoryTheory.Functor (Opposite C) A
    s : CategoryTheory.Functor A (Type (max v₁ u₁))
    inst✝ : ∀ (J : Type (max v₁ u₁)), CategoryTheory.Limits.PreservesLimitsOfShape …
    U : C
    R : CategoryTheory.Presieve U
    e : CategoryTheory.Iso (CategoryTheory.Limits.parallelPair (s.map (CategoryThe …
    ⊢ Equiv (CategoryTheory.Limits.IsLimit (s.mapCone (CategoryTheory.Limits.Fork. …
  -/
  refine Equiv.trans (isLimitMapConeForkEquiv _ _) ?_
  refine (IsLimit.postcomposeHomEquiv e _).symm.trans
    (IsLimit.equivIsoLimit (Fork.ext (Iso.refl _) ?_))
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₂, u₂} A
    A' : Type u₂
    inst✝⁵ : CategoryTheory.Category.{max v₁ u₁, u₂} A'
    B : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₃, u₃} B
    J : CategoryTheory.GrothendieckTopology C
    U✝ : C
    R✝ : CategoryTheory.Presieve U✝
    P✝ : CategoryTheory.Functor (Opposite C) A
    P' : CategoryTheory.Functor (Opposite C) A'
    inst✝³ : CategoryTheory.Limits.HasProducts A
    inst✝² : CategoryTheory.Limits.HasProducts A'
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    P : CategoryTheory.Functor (Opposite C) A
    s : CategoryTheory.Functor A (Type (max v₁ u₁))
    inst✝ : ∀ (J : Type (max v₁ u₁)), CategoryTheory.Limits.PreservesLimitsOfShape …
    U : C
    R : CategoryTheory.Presieve U
    e : CategoryTheory.Iso (CategoryTheory.Limits.parallelPair (s.map (CategoryThe …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.refl ((CategoryTh …
  -/
  dsimp [Equalizer.forkMap, forkMap, e, Fork.ι]
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₂, u₂} A
    A' : Type u₂
    inst✝⁵ : CategoryTheory.Category.{max v₁ u₁, u₂} A'
    B : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₃, u₃} B
    J : CategoryTheory.GrothendieckTopology C
    U✝ : C
    R✝ : CategoryTheory.Presieve U✝
    P✝ : CategoryTheory.Functor (Opposite C) A
    P' : CategoryTheory.Functor (Opposite C) A'
    inst✝³ : CategoryTheory.Limits.HasProducts A
    inst✝² : CategoryTheory.Limits.HasProducts A'
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    P : CategoryTheory.Functor (Opposite C) A
    s : CategoryTheory.Functor A (Type (max v₁ u₁))
    inst✝ : ∀ (J : Type (max v₁ u₁)), CategoryTheory.Limits.PreservesLimitsOfShape …
    U : C
    R : CategoryTheory.Presieve U
    e : CategoryTheory.Iso (CategoryTheory.Limits.parallelPair (s.map (CategoryThe …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (s. …
  -/
  simp only [id_comp, map_lift_piComparison]
  /-
    🎉 no goals
  -/

-- Remark : this lemma uses `A'` not `A`; `A'` is `A` but with a universe
-- restriction. Can it be generalised?

/-- The equalizer definition of a sheaf given by `isSheaf'` is equivalent to `isSheaf`. -/
theorem isSheaf_iff_isSheaf' : IsSheaf J P' ↔ IsSheaf' J P' := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    A' : Type u₂
    inst✝² : CategoryTheory.Category.{max v₁ u₁, u₂} A'
    J : CategoryTheory.GrothendieckTopology C
    P' : CategoryTheory.Functor (Opposite C) A'
    inst✝¹ : CategoryTheory.Limits.HasProducts A'
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    ⊢ Iff (CategoryTheory.Presheaf.IsSheaf J P') (CategoryTheory.Presheaf.IsSheaf' …
  -/
  constructor
    /-
      case mp
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      A' : Type u₂
      inst✝² : CategoryTheory.Category.{max v₁ u₁, u₂} A'
      J : CategoryTheory.GrothendieckTopology C
      P' : CategoryTheory.Functor (Opposite C) A'
      inst✝¹ : CategoryTheory.Limits.HasProducts A'
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      ⊢ CategoryTheory.Presheaf.IsSheaf J P' → CategoryTheory.Presheaf.IsSheaf' J P'
    -/
  · intro h U R hR
    /-
      case mp
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      A' : Type u₂
      inst✝² : CategoryTheory.Category.{max v₁ u₁, u₂} A'
      J : CategoryTheory.GrothendieckTopology C
      P' : CategoryTheory.Functor (Opposite C) A'
      inst✝¹ : CategoryTheory.Limits.HasProducts A'
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      h : CategoryTheory.Presheaf.IsSheaf J P'
      U : C
      R : CategoryTheory.Presieve U
      hR : Membership.mem (J U) (CategoryTheory.Sieve.generate R)
      ⊢ Nonempty (CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι (Cat …
    -/
    refine ⟨?_⟩
    /-
      case mp
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      A' : Type u₂
      inst✝² : CategoryTheory.Category.{max v₁ u₁, u₂} A'
      J : CategoryTheory.GrothendieckTopology C
      P' : CategoryTheory.Functor (Opposite C) A'
      inst✝¹ : CategoryTheory.Limits.HasProducts A'
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      h : CategoryTheory.Presheaf.IsSheaf J P'
      U : C
      R : CategoryTheory.Presieve U
      hR : Membership.mem (J U) (CategoryTheory.Sieve.generate R)
      ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι (CategoryTheor …
    -/
    apply coyonedaJointlyReflectsLimits
    /-
      case mp.hc
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      A' : Type u₂
      inst✝² : CategoryTheory.Category.{max v₁ u₁, u₂} A'
      J : CategoryTheory.GrothendieckTopology C
      P' : CategoryTheory.Functor (Opposite C) A'
      inst✝¹ : CategoryTheory.Limits.HasProducts A'
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      h : CategoryTheory.Presheaf.IsSheaf J P'
      U : C
      R : CategoryTheory.Presieve U
      hR : Membership.mem (J U) (CategoryTheory.Sieve.generate R)
      ⊢ (X : Opposite A') → CategoryTheory.Limits.IsLimit ((CategoryTheory.coyoneda. …
    -/
    intro X
    /-
      case mp.hc
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      A' : Type u₂
      inst✝² : CategoryTheory.Category.{max v₁ u₁, u₂} A'
      J : CategoryTheory.GrothendieckTopology C
      P' : CategoryTheory.Functor (Opposite C) A'
      inst✝¹ : CategoryTheory.Limits.HasProducts A'
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      h : CategoryTheory.Presheaf.IsSheaf J P'
      U : C
      R : CategoryTheory.Presieve U
      hR : Membership.mem (J U) (CategoryTheory.Sieve.generate R)
      X : Opposite A'
      ⊢ CategoryTheory.Limits.IsLimit ((CategoryTheory.coyoneda.obj X).mapCone (Cate …
    -/
    have q : Presieve.IsSheafFor (P' ⋙ coyoneda.obj X) _ := h X.unop _ hR
    /-
      case mp.hc
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      A' : Type u₂
      inst✝² : CategoryTheory.Category.{max v₁ u₁, u₂} A'
      J : CategoryTheory.GrothendieckTopology C
      P' : CategoryTheory.Functor (Opposite C) A'
      inst✝¹ : CategoryTheory.Limits.HasProducts A'
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      h : CategoryTheory.Presheaf.IsSheaf J P'
      U : C
      R : CategoryTheory.Presieve U
      hR : Membership.mem (J U) (CategoryTheory.Sieve.generate R)
      X : Opposite A'
      q : CategoryTheory.Presieve.IsSheafFor (P'.comp (CategoryTheory.coyoneda.obj X …
      ⊢ CategoryTheory.Limits.IsLimit ((CategoryTheory.coyoneda.obj X).mapCone (Cate …
    -/
    rw [← Presieve.isSheafFor_iff_generate] at q
    /-
      case mp.hc
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      A' : Type u₂
      inst✝² : CategoryTheory.Category.{max v₁ u₁, u₂} A'
      J : CategoryTheory.GrothendieckTopology C
      P' : CategoryTheory.Functor (Opposite C) A'
      inst✝¹ : CategoryTheory.Limits.HasProducts A'
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      h : CategoryTheory.Presheaf.IsSheaf J P'
      U : C
      R : CategoryTheory.Presieve U
      hR : Membership.mem (J U) (CategoryTheory.Sieve.generate R)
      X : Opposite A'
      q : CategoryTheory.Presieve.IsSheafFor (P'.comp (CategoryTheory.coyoneda.obj X …
      ⊢ CategoryTheory.Limits.IsLimit ((CategoryTheory.coyoneda.obj X).mapCone (Cate …
    -/
    rw [Equalizer.Presieve.sheaf_condition] at q
    /-
      case mp.hc
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      A' : Type u₂
      inst✝² : CategoryTheory.Category.{max v₁ u₁, u₂} A'
      J : CategoryTheory.GrothendieckTopology C
      P' : CategoryTheory.Functor (Opposite C) A'
      inst✝¹ : CategoryTheory.Limits.HasProducts A'
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      h : CategoryTheory.Presheaf.IsSheaf J P'
      U : C
      R : CategoryTheory.Presieve U
      hR : Membership.mem (J U) (CategoryTheory.Sieve.generate R)
      X : Opposite A'
      q : Nonempty (CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι (C …
      ⊢ CategoryTheory.Limits.IsLimit ((CategoryTheory.coyoneda.obj X).mapCone (Cate …
    -/
    replace q := Classical.choice q
    /-
      case mp.hc
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      A' : Type u₂
      inst✝² : CategoryTheory.Category.{max v₁ u₁, u₂} A'
      J : CategoryTheory.GrothendieckTopology C
      P' : CategoryTheory.Functor (Opposite C) A'
      inst✝¹ : CategoryTheory.Limits.HasProducts A'
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      h : CategoryTheory.Presheaf.IsSheaf J P'
      U : C
      R : CategoryTheory.Presieve U
      hR : Membership.mem (J U) (CategoryTheory.Sieve.generate R)
      X : Opposite A'
      q : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι (CategoryThe …
      ⊢ CategoryTheory.Limits.IsLimit ((CategoryTheory.coyoneda.obj X).mapCone (Cate …
    -/
    apply (isSheafForIsSheafFor' _ _ _ _).symm q
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      A' : Type u₂
      inst✝² : CategoryTheory.Category.{max v₁ u₁, u₂} A'
      J : CategoryTheory.GrothendieckTopology C
      P' : CategoryTheory.Functor (Opposite C) A'
      inst✝¹ : CategoryTheory.Limits.HasProducts A'
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      ⊢ CategoryTheory.Presheaf.IsSheaf' J P' → CategoryTheory.Presheaf.IsSheaf J P'
    -/
  · intro h U X S hS
    /-
      case mpr
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      A' : Type u₂
      inst✝² : CategoryTheory.Category.{max v₁ u₁, u₂} A'
      J : CategoryTheory.GrothendieckTopology C
      P' : CategoryTheory.Functor (Opposite C) A'
      inst✝¹ : CategoryTheory.Limits.HasProducts A'
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      h : CategoryTheory.Presheaf.IsSheaf' J P'
      U : A'
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J X) S
      ⊢ CategoryTheory.Presieve.IsSheafFor (P'.comp (CategoryTheory.coyoneda.obj { u …
    -/
    rw [Equalizer.Presieve.sheaf_condition]
    /-
      case mpr
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      A' : Type u₂
      inst✝² : CategoryTheory.Category.{max v₁ u₁, u₂} A'
      J : CategoryTheory.GrothendieckTopology C
      P' : CategoryTheory.Functor (Opposite C) A'
      inst✝¹ : CategoryTheory.Limits.HasProducts A'
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      h : CategoryTheory.Presheaf.IsSheaf' J P'
      U : A'
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J X) S
      ⊢ Nonempty (CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι (Cat …
    -/
    refine ⟨?_⟩
    /-
      case mpr
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      A' : Type u₂
      inst✝² : CategoryTheory.Category.{max v₁ u₁, u₂} A'
      J : CategoryTheory.GrothendieckTopology C
      P' : CategoryTheory.Functor (Opposite C) A'
      inst✝¹ : CategoryTheory.Limits.HasProducts A'
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      h : CategoryTheory.Presheaf.IsSheaf' J P'
      U : A'
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J X) S
      ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι (CategoryTheor …
    -/
    refine isSheafForIsSheafFor' _ _ _ _ ?_
    /-
      case mpr
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      A' : Type u₂
      inst✝² : CategoryTheory.Category.{max v₁ u₁, u₂} A'
      J : CategoryTheory.GrothendieckTopology C
      P' : CategoryTheory.Functor (Opposite C) A'
      inst✝¹ : CategoryTheory.Limits.HasProducts A'
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      h : CategoryTheory.Presheaf.IsSheaf' J P'
      U : A'
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J X) S
      ⊢ CategoryTheory.Limits.IsLimit ((CategoryTheory.coyoneda.obj { unop := U }).m …
    -/
    letI := preservesSmallestLimits_of_preservesLimits (coyoneda.obj (op U))
    /-
      case mpr
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      A' : Type u₂
      inst✝² : CategoryTheory.Category.{max v₁ u₁, u₂} A'
      J : CategoryTheory.GrothendieckTopology C
      P' : CategoryTheory.Functor (Opposite C) A'
      inst✝¹ : CategoryTheory.Limits.HasProducts A'
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      h : CategoryTheory.Presheaf.IsSheaf' J P'
      U : A'
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J X) S
      this : CategoryTheory.Limits.PreservesLimitsOfSize.{0, 0, max u₁ v₁, max u₁ v₁ …
      ⊢ CategoryTheory.Limits.IsLimit ((CategoryTheory.coyoneda.obj { unop := U }).m …
    -/
    apply isLimitOfPreserves
    /-
      case mpr.t
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      A' : Type u₂
      inst✝² : CategoryTheory.Category.{max v₁ u₁, u₂} A'
      J : CategoryTheory.GrothendieckTopology C
      P' : CategoryTheory.Functor (Opposite C) A'
      inst✝¹ : CategoryTheory.Limits.HasProducts A'
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      h : CategoryTheory.Presheaf.IsSheaf' J P'
      U : A'
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J X) S
      this : CategoryTheory.Limits.PreservesLimitsOfSize.{0, 0, max u₁ v₁, max u₁ v₁ …
      ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι (CategoryTheor …
    -/
    apply Classical.choice (h _ S.arrows _)
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      A' : Type u₂
      inst✝² : CategoryTheory.Category.{max v₁ u₁, u₂} A'
      J : CategoryTheory.GrothendieckTopology C
      P' : CategoryTheory.Functor (Opposite C) A'
      inst✝¹ : CategoryTheory.Limits.HasProducts A'
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      h : CategoryTheory.Presheaf.IsSheaf' J P'
      U : A'
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J X) S
      this : CategoryTheory.Limits.PreservesLimitsOfSize.{0, 0, max u₁ v₁, max u₁ v₁ …
      ⊢ Membership.mem (J X) (CategoryTheory.Sieve.generate S.arrows)
    -/
    simpa
    /-
      🎉 no goals
    -/


theorem isSheaf_of_isSheaf_comp (s : A ⥤ B) [ReflectsLimitsOfSize.{v₁, max v₁ u₁} s]
    (h : IsSheaf J (P ⋙ s)) : IsSheaf J P := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} A
    B : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} B
    J : CategoryTheory.GrothendieckTopology C
    P : CategoryTheory.Functor (Opposite C) A
    s : CategoryTheory.Functor A B
    inst✝ : CategoryTheory.Limits.ReflectsLimitsOfSize.{v₁, max v₁ u₁, v₂, v₃, u₂, …
    h : CategoryTheory.Presheaf.IsSheaf J (P.comp s)
    ⊢ CategoryTheory.Presheaf.IsSheaf J P
  -/
  rw [isSheaf_iff_isLimit] at h ⊢
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} A
    B : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} B
    J : CategoryTheory.GrothendieckTopology C
    P : CategoryTheory.Functor (Opposite C) A
    s : CategoryTheory.Functor A B
    inst✝ : CategoryTheory.Limits.ReflectsLimitsOfSize.{v₁, max v₁ u₁, v₂, v₃, u₂, …
    h : ∀ ⦃X : C⦄ (S : CategoryTheory.Sieve X), Membership.mem (J X) S → Nonempty  …
    ⊢ ∀ ⦃X : C⦄ (S : CategoryTheory.Sieve X), Membership.mem (J X) S → Nonempty (C …
  -/
  exact fun X S hS ↦ (h S hS).map fun t ↦ isLimitOfReflects s t
  /-
    🎉 no goals
  -/


theorem isSheaf_comp_of_isSheaf (s : A ⥤ B) [PreservesLimitsOfSize.{v₁, max v₁ u₁} s]
    (h : IsSheaf J P) : IsSheaf J (P ⋙ s) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} A
    B : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} B
    J : CategoryTheory.GrothendieckTopology C
    P : CategoryTheory.Functor (Opposite C) A
    s : CategoryTheory.Functor A B
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfSize.{v₁, max v₁ u₁, v₂, v₃, u₂ …
    h : CategoryTheory.Presheaf.IsSheaf J P
    ⊢ CategoryTheory.Presheaf.IsSheaf J (P.comp s)
  -/
  rw [isSheaf_iff_isLimit] at h ⊢
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} A
    B : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} B
    J : CategoryTheory.GrothendieckTopology C
    P : CategoryTheory.Functor (Opposite C) A
    s : CategoryTheory.Functor A B
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfSize.{v₁, max v₁ u₁, v₂, v₃, u₂ …
    h : ∀ ⦃X : C⦄ (S : CategoryTheory.Sieve X), Membership.mem (J X) S → Nonempty  …
    ⊢ ∀ ⦃X : C⦄ (S : CategoryTheory.Sieve X), Membership.mem (J X) S → Nonempty (C …
  -/
  apply fun X S hS ↦ (h S hS).map fun t ↦ isLimitOfPreserves s t
  /-
    🎉 no goals
  -/


theorem isSheaf_iff_isSheaf_comp (s : A ⥤ B) [HasLimitsOfSize.{v₁, max v₁ u₁} A]
    [PreservesLimitsOfSize.{v₁, max v₁ u₁} s] [s.ReflectsIsomorphisms] :
    IsSheaf J P ↔ IsSheaf J (P ⋙ s) := by
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} A
    B : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} B
    J : CategoryTheory.GrothendieckTopology C
    P : CategoryTheory.Functor (Opposite C) A
    s : CategoryTheory.Functor A B
    inst✝² : CategoryTheory.Limits.HasLimitsOfSize.{v₁, max v₁ u₁, v₂, u₂} A
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfSize.{v₁, max v₁ u₁, v₂, v₃, u …
    inst✝ : s.ReflectsIsomorphisms
    ⊢ Iff (CategoryTheory.Presheaf.IsSheaf J P) (CategoryTheory.Presheaf.IsSheaf J …
  -/
  letI : ReflectsLimitsOfSize s := reflectsLimits_of_reflectsIsomorphisms
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} A
    B : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} B
    J : CategoryTheory.GrothendieckTopology C
    P : CategoryTheory.Functor (Opposite C) A
    s : CategoryTheory.Functor A B
    inst✝² : CategoryTheory.Limits.HasLimitsOfSize.{v₁, max v₁ u₁, v₂, u₂} A
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfSize.{v₁, max v₁ u₁, v₂, v₃, u …
    inst✝ : s.ReflectsIsomorphisms
    this : CategoryTheory.Limits.ReflectsLimitsOfSize.{v₁, max u₁ v₁, v₂, v₃, u₂,  …
    ⊢ Iff (CategoryTheory.Presheaf.IsSheaf J P) (CategoryTheory.Presheaf.IsSheaf J …
  -/
  exact ⟨isSheaf_comp_of_isSheaf J P s, isSheaf_of_isSheaf_comp J P s⟩
  /-
    🎉 no goals
  -/


/--
For a concrete category `(A, s)` where the forgetful functor `s : A ⥤ Type v` preserves limits and
reflects isomorphisms, and `A` has limits, an `A`-valued presheaf `P : Cᵒᵖ ⥤ A` is a sheaf iff its
underlying `Type`-valued presheaf `P ⋙ s : Cᵒᵖ ⥤ Type` is a sheaf.

Note this lemma applies for "algebraic" categories, eg groups, abelian groups and rings, but not
for the category of topological spaces, topological rings, etc since reflecting isomorphisms doesn't
hold.
-/
theorem isSheaf_iff_isSheaf_forget (s : A' ⥤ Type max v₁ u₁) [HasLimits A'] [PreservesLimits s]
    [s.ReflectsIsomorphisms] : IsSheaf J P' ↔ IsSheaf J (P' ⋙ s) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    A' : Type u₂
    inst✝³ : CategoryTheory.Category.{max v₁ u₁, u₂} A'
    J : CategoryTheory.GrothendieckTopology C
    P' : CategoryTheory.Functor (Opposite C) A'
    s : CategoryTheory.Functor A' (Type (max v₁ u₁))
    inst✝² : CategoryTheory.Limits.HasLimits A'
    inst✝¹ : CategoryTheory.Limits.PreservesLimits s
    inst✝ : s.ReflectsIsomorphisms
    ⊢ Iff (CategoryTheory.Presheaf.IsSheaf J P') (CategoryTheory.Presheaf.IsSheaf  …
  -/
  have : HasLimitsOfSize.{v₁, max v₁ u₁} A' := hasLimitsOfSizeShrink.{_, _, u₁, 0} A'
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    A' : Type u₂
    inst✝³ : CategoryTheory.Category.{max v₁ u₁, u₂} A'
    J : CategoryTheory.GrothendieckTopology C
    P' : CategoryTheory.Functor (Opposite C) A'
    s : CategoryTheory.Functor A' (Type (max v₁ u₁))
    inst✝² : CategoryTheory.Limits.HasLimits A'
    inst✝¹ : CategoryTheory.Limits.PreservesLimits s
    inst✝ : s.ReflectsIsomorphisms
    this : CategoryTheory.Limits.HasLimitsOfSize.{v₁, max v₁ u₁, max u₁ v₁, u₂} A'
    ⊢ Iff (CategoryTheory.Presheaf.IsSheaf J P') (CategoryTheory.Presheaf.IsSheaf  …
  -/
  have : PreservesLimitsOfSize.{v₁, max v₁ u₁} s := preservesLimitsOfSize_shrink.{_, 0, _, u₁} s
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    A' : Type u₂
    inst✝³ : CategoryTheory.Category.{max v₁ u₁, u₂} A'
    J : CategoryTheory.GrothendieckTopology C
    P' : CategoryTheory.Functor (Opposite C) A'
    s : CategoryTheory.Functor A' (Type (max v₁ u₁))
    inst✝² : CategoryTheory.Limits.HasLimits A'
    inst✝¹ : CategoryTheory.Limits.PreservesLimits s
    inst✝ : s.ReflectsIsomorphisms
    this✝ : CategoryTheory.Limits.HasLimitsOfSize.{v₁, max v₁ u₁, max u₁ v₁, u₂} A'
    this : CategoryTheory.Limits.PreservesLimitsOfSize.{v₁, max v₁ u₁, max u₁ v₁,  …
    ⊢ Iff (CategoryTheory.Presheaf.IsSheaf J P') (CategoryTheory.Presheaf.IsSheaf  …
  -/
  apply isSheaf_iff_isSheaf_comp
  /-
    🎉 no goals
  -/


