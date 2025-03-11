/-- The residue field of `X` at a point `x` is the residue field of the stalk of `X`
at `x`. -/
def residueField (x : X) : CommRingCat :=
  CommRingCat.of <| IsLocalRing.ResidueField (X.presheaf.stalk x)


instance (x : X) : Field (X.residueField x) :=
  inferInstanceAs <| Field (IsLocalRing.ResidueField (X.presheaf.stalk x))


/--
If `U` is an open of `X` containing `x`, we have a canonical ring map from the sections
over `U` to the residue field of `x`.

If we interpret sections over `U` as functions of `X` defined on `U`, then this ring map
corresponds to evaluation at `x`.
-/
def evaluation (x : U) : X.presheaf.obj (op U) ⟶ X.residueField x :=
  -- TODO: make a new definition wrapping
  -- `CommRingCat.ofHom (IsLocalRing.residue (X.presheaf.stalk _))`?
  X.presheaf.germ U x.1 x.2 ≫ CommRingCat.ofHom (IsLocalRing.residue (X.presheaf.stalk _))


/-- The global evaluation map from `Γ(X, ⊤)` to the residue field at `x`. -/
def Γevaluation (x : X) : X.presheaf.obj (op ⊤) ⟶ X.residueField x :=
  X.evaluation ⟨x, show x ∈ ⊤ from trivial⟩


@[simp]
lemma evaluation_eq_zero_iff_not_mem_basicOpen (x : U) (f : X.presheaf.obj (op U)) :
    X.evaluation x f = 0 ↔ x.val ∉ X.toRingedSpace.basicOpen f := by
  /-
    X : AlgebraicGeometry.LocallyRingedSpace
    U : TopologicalSpace.Opens ↑X.toTopCat
    x : Subtype fun x => Membership.mem U x
    f : ↑(X.presheaf.obj { unop := U })
    ⊢ Iff (Eq ((X.evaluation x).hom f) 0) (Not (Membership.mem (X.toRingedSpace.ba …
  -/
  rw [X.toRingedSpace.mem_basicOpen f x.1 x.2, ← not_iff_not, not_not]
  /-
    X : AlgebraicGeometry.LocallyRingedSpace
    U : TopologicalSpace.Opens ↑X.toTopCat
    x : Subtype fun x => Membership.mem U x
    f : ↑(X.presheaf.obj { unop := U })
    ⊢ Iff (Not (Eq ((X.evaluation x).hom f) 0)) (IsUnit ((X.toRingedSpace.presheaf …
  -/
  exact (IsLocalRing.residue_ne_zero_iff_isUnit _)
  /-
    🎉 no goals
  -/


lemma evaluation_ne_zero_iff_mem_basicOpen (x : U) (f : X.presheaf.obj (op U)) :
    X.evaluation x f ≠ 0 ↔ x.val ∈ X.toRingedSpace.basicOpen f := by
  /-
    X : AlgebraicGeometry.LocallyRingedSpace
    U : TopologicalSpace.Opens ↑X.toTopCat
    x : Subtype fun x => Membership.mem U x
    f : ↑(X.presheaf.obj { unop := U })
    ⊢ Iff (Ne ((X.evaluation x).hom f) 0) (Membership.mem (X.toRingedSpace.basicOp …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma basicOpen_eq_bot_iff_forall_evaluation_eq_zero (f : X.presheaf.obj (op U)) :
    X.toRingedSpace.basicOpen f = ⊥ ↔ ∀ (x : U), X.evaluation x f = 0 := by
  /-
    X : AlgebraicGeometry.LocallyRingedSpace
    U : TopologicalSpace.Opens ↑X.toTopCat
    f : ↑(X.presheaf.obj { unop := U })
    ⊢ Iff (Eq (X.toRingedSpace.basicOpen f) Bot.bot) (∀ (x : Subtype fun x => Memb …
  -/
  simp only [evaluation_eq_zero_iff_not_mem_basicOpen, Subtype.forall]
  exact ⟨fun h ↦ h ▸ fun a _ hc ↦ hc,
    fun h ↦ eq_bot_iff.mpr <| fun a ha ↦ h a (X.toRingedSpace.basicOpen_le f ha) ha⟩


@[simp]
lemma Γevaluation_eq_zero_iff_not_mem_basicOpen (x : X) (f : X.presheaf.obj (op ⊤)) :
    X.Γevaluation x f = 0 ↔ x ∉ X.toRingedSpace.basicOpen f :=
                                                               /-
                                                                 X : AlgebraicGeometry.LocallyRingedSpace
                                                                 x : ↑X.toTopCat
                                                                 f : ↑(X.presheaf.obj { unop := Top.top })
                                                                 ⊢ Membership.mem Top.top x
                                                               -/
  evaluation_eq_zero_iff_not_mem_basicOpen X ⟨x, show x ∈ ⊤ by trivial⟩ f
                                                               /-
                                                                 🎉 no goals
                                                               -/


lemma Γevaluation_ne_zero_iff_mem_basicOpen (x : X) (f : X.presheaf.obj (op ⊤)) :
    X.Γevaluation x f ≠ 0 ↔ x ∈ X.toRingedSpace.basicOpen f :=
                                                           /-
                                                             X : AlgebraicGeometry.LocallyRingedSpace
                                                             x : ↑X.toTopCat
                                                             f : ↑(X.presheaf.obj { unop := Top.top })
                                                             ⊢ Membership.mem Top.top x
                                                           -/
  evaluation_ne_zero_iff_mem_basicOpen X ⟨x, show x ∈ ⊤ by trivial⟩ f
                                                           /-
                                                             🎉 no goals
                                                           -/


/-- If `X ⟶ Y` is a morphism of locally ringed spaces and `x` a point of `X`, we obtain
a morphism of residue fields in the other direction. -/
def residueFieldMap (x : X) : Y.residueField (f.base x) ⟶ X.residueField x :=
  CommRingCat.ofHom (IsLocalRing.ResidueField.map (f.stalkMap x).hom)


lemma residue_comp_residueFieldMap_eq_stalkMap_comp_residue (x : X) :
    CommRingCat.ofHom (IsLocalRing.residue (Y.presheaf.stalk (f.base x))) ≫
      residueFieldMap f x = f.stalkMap x ≫
      CommRingCat.ofHom (IsLocalRing.residue (X.presheaf.stalk x)) := by
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Y
    x : ↑X.toTopCat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom (IsLocalRing.resid …
  -/
  simp [residueFieldMap]
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Y
    x : ↑X.toTopCat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom (IsLocalRing.resid …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma residueFieldMap_id (x : X) :
    residueFieldMap (𝟙 X) x = 𝟙 (X.residueField x) := by
  /-
    X : AlgebraicGeometry.LocallyRingedSpace
    x : ↑X.toTopCat
    ⊢ Eq (AlgebraicGeometry.LocallyRingedSpace.residueFieldMap (CategoryTheory.Cat …
  -/
  ext : 1
  /-
    case hf
    X : AlgebraicGeometry.LocallyRingedSpace
    x : ↑X.toTopCat
    ⊢ Eq (AlgebraicGeometry.LocallyRingedSpace.residueFieldMap (CategoryTheory.Cat …
  -/
  simp only [id_toShHom', SheafedSpace.id_base, TopCat.coe_id, id_eq, residueFieldMap, stalkMap_id]
  /-
    case hf
    X : AlgebraicGeometry.LocallyRingedSpace
    x : ↑X.toTopCat
    ⊢ Eq (IsLocalRing.ResidueField.map (CategoryTheory.CategoryStruct.id (X.preshe …
  -/
  apply IsLocalRing.ResidueField.map_id
  /-
    🎉 no goals
  -/


@[simp]
lemma residueFieldMap_comp {Z : LocallyRingedSpace.{u}} (g : Y ⟶ Z) (x : X) :
    residueFieldMap (f ≫ g) x = residueFieldMap g (f.base x) ≫ residueFieldMap f x := by
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Y
    Z : AlgebraicGeometry.LocallyRingedSpace
    g : Quiver.Hom Y Z
    x : ↑X.toTopCat
    ⊢ Eq (AlgebraicGeometry.LocallyRingedSpace.residueFieldMap (CategoryTheory.Cat …
  -/
  ext : 1
  simp only [comp_toShHom, SheafedSpace.comp_base, Function.comp_apply, residueFieldMap,
    CommRingCat.hom_comp, TopCat.comp_app]
  /-
    case hf
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Y
    Z : AlgebraicGeometry.LocallyRingedSpace
    g : Quiver.Hom Y Z
    x : ↑X.toTopCat
    ⊢ Eq (IsLocalRing.ResidueField.map (AlgebraicGeometry.LocallyRingedSpace.Hom.s …
  -/
  simp_rw [stalkMap_comp]
  /-
    case hf
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Y
    Z : AlgebraicGeometry.LocallyRingedSpace
    g : Quiver.Hom Y Z
    x : ↑X.toTopCat
    ⊢ Eq (IsLocalRing.ResidueField.map (CategoryTheory.CategoryStruct.comp (Algebr …
  -/
  apply IsLocalRing.ResidueField.map_comp
  /-
    🎉 no goals
  -/


@[reassoc]
lemma evaluation_naturality {V : Opens Y} (x : (Opens.map f.base).obj V) :
    Y.evaluation ⟨f.base x, x.property⟩ ≫ residueFieldMap f x.val =
      f.c.app (op V) ≫ X.evaluation x := by
  dsimp only [LocallyRingedSpace.evaluation,
    LocallyRingedSpace.residueFieldMap]
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Y
    V : TopologicalSpace.Opens ↑Y.toTopCat
    x : Subtype fun x => Membership.mem ((TopologicalSpace.Opens.map f.base).obj V …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [Category.assoc]
  /-
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Y
    V : TopologicalSpace.Opens ↑Y.toTopCat
    x : Subtype fun x => Membership.mem ((TopologicalSpace.Opens.map f.base).obj V …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Y.presheaf.germ V (f.base ↑x) ⋯) (Ca …
  -/
  ext a
  /-
    case hf.a
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Y
    V : TopologicalSpace.Opens ↑Y.toTopCat
    x : Subtype fun x => Membership.mem ((TopologicalSpace.Opens.map f.base).obj V …
    a : ↑(Y.presheaf.obj { unop := V })
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (Y.presheaf.germ V (f.base ↑x) ⋯) (C …
  -/
  simp only [CommRingCat.comp_apply]
  /-
    case hf.a
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Y
    V : TopologicalSpace.Opens ↑Y.toTopCat
    x : Subtype fun x => Membership.mem ((TopologicalSpace.Opens.map f.base).obj V …
    a : ↑(Y.presheaf.obj { unop := V })
    ⊢ Eq ((IsLocalRing.ResidueField.map (AlgebraicGeometry.LocallyRingedSpace.Hom. …
  -/
  erw [IsLocalRing.ResidueField.map_residue]
  /-
    case hf.a
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Y
    V : TopologicalSpace.Opens ↑Y.toTopCat
    x : Subtype fun x => Membership.mem ((TopologicalSpace.Opens.map f.base).obj V …
    a : ↑(Y.presheaf.obj { unop := V })
    ⊢ Eq ((IsLocalRing.residue ↑(X.presheaf.stalk ↑x)) ((AlgebraicGeometry.Locally …
  -/
  rw [LocallyRingedSpace.stalkMap_germ_apply]
  /-
    case hf.a
    X Y : AlgebraicGeometry.LocallyRingedSpace
    f : Quiver.Hom X Y
    V : TopologicalSpace.Opens ↑Y.toTopCat
    x : Subtype fun x => Membership.mem ((TopologicalSpace.Opens.map f.base).obj V …
    a : ↑(Y.presheaf.obj { unop := V })
    ⊢ Eq ((IsLocalRing.residue ↑(X.presheaf.stalk ↑x)) ((X.presheaf.germ ((Topolog …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma evaluation_naturality_apply {V : Opens Y} (x : (Opens.map f.base).obj V)
    (a : Y.presheaf.obj (op V)) :
    residueFieldMap f x.val (Y.evaluation ⟨f.base x, x.property⟩ a) =
      X.evaluation x (f.c.app (op V) a) := by
  simpa using congrFun (congrArg (DFunLike.coe ∘ CommRingCat.Hom.hom) <|
    evaluation_naturality f x) a


@[reassoc]
lemma Γevaluation_naturality (x : X) :
    Y.Γevaluation (f.base x) ≫ residueFieldMap f x =
      f.c.app (op ⊤) ≫ X.Γevaluation x :=
                                 /-
                                   X Y : AlgebraicGeometry.LocallyRingedSpace
                                   f : Quiver.Hom X Y
                                   x : ↑X.toTopCat
                                   ⊢ Membership.mem ((TopologicalSpace.Opens.map f.base).obj Top.top) x
                                 -/
  evaluation_naturality f ⟨x, by simp only [Opens.map_top]; trivial⟩
                                                            /-
                                                              🎉 no goals
                                                            -/


lemma Γevaluation_naturality_apply (x : X) (a : Y.presheaf.obj (op ⊤)) :
    residueFieldMap f x (Y.Γevaluation (f.base x) a) =
      X.Γevaluation x (f.c.app (op ⊤) a) :=
                                       /-
                                         X Y : AlgebraicGeometry.LocallyRingedSpace
                                         f : Quiver.Hom X Y
                                         x : ↑X.toTopCat
                                         a : ↑(Y.presheaf.obj { unop := Top.top })
                                         ⊢ Membership.mem ((TopologicalSpace.Opens.map f.base).obj Top.top) x
                                       -/
  evaluation_naturality_apply f ⟨x, by simp only [Opens.map_top]; trivial⟩ a
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


