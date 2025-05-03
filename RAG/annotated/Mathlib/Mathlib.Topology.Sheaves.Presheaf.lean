/-- The category of `C`-valued presheaves on a (bundled) topological space `X`. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): was @[nolint has_nonempty_instance]
def Presheaf (X : TopCat.{w}) : Type max u v w :=
  (Opens X)ᵒᵖ ⥤ C


instance (X : TopCat.{w}) : Category (Presheaf.{w, v, u} C X) :=
  inferInstanceAs (Category ((Opens X)ᵒᵖ ⥤ C : Type max u v w))


@[simp] theorem comp_app {X : TopCat} {U : (Opens X)ᵒᵖ} {P Q R : Presheaf C X}
    (f : P ⟶ Q) (g : Q ⟶ R) :
    (f ≫ g).app U = f.app U ≫ g.app U := rfl


@[ext]
lemma ext {X : TopCat} {P Q : Presheaf C X} {f g : P ⟶ Q}
    (w : ∀ U : Opens X, f.app (op U) = g.app (op U)) :
    f = g := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    P Q : TopCat.Presheaf C X
    f g : Quiver.Hom P Q
    w : ∀ (U : TopologicalSpace.Opens ↑X), Eq (f.app { unop := U }) (g.app { unop  …
    ⊢ Eq f g
  -/
  apply NatTrans.ext
  /-
    case app
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    P Q : TopCat.Presheaf C X
    f g : Quiver.Hom P Q
    w : ∀ (U : TopologicalSpace.Opens ↑X), Eq (f.app { unop := U }) (g.app { unop  …
    ⊢ Eq f.app g.app
  -/
  ext U
  /-
    case app.h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    P Q : TopCat.Presheaf C X
    f g : Quiver.Hom P Q
    w : ∀ (U : TopologicalSpace.Opens ↑X), Eq (f.app { unop := U }) (g.app { unop  …
    U : Opposite (TopologicalSpace.Opens ↑X)
    ⊢ Eq (f.app U) (g.app U)
  -/
  induction U with | _ U => ?_
  /-
    case app.h.h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    P Q : TopCat.Presheaf C X
    f g : Quiver.Hom P Q
    w : ∀ (U : TopologicalSpace.Opens ↑X), Eq (f.app { unop := U }) (g.app { unop  …
    U : TopologicalSpace.Opens ↑X
    ⊢ Eq (f.app { unop := U }) (g.app { unop := U })
  -/
  apply w
  /-
    🎉 no goals
  -/


/-- attribute `sheaf_restrict` to mark lemmas related to restricting sheaves -/
macro "sheaf_restrict" : attr =>
  `(attr|aesop safe 50 apply (rule_sets := [$(Lean.mkIdent `Restrict):ident]))


/-- `restrict_tac` solves relations among subsets (copied from `aesop cat`) -/
macro (name := restrict_tac) "restrict_tac" c:Aesop.tactic_clause* : tactic =>
`(tactic| first | assumption |
  aesop $c*
    (config := { terminal := true
                 assumptionTransparency := .reducible
                 enableSimp := false })
    (rule_sets := [-default, -builtin, $(Lean.mkIdent `Restrict):ident]))


/-- `restrict_tac?` passes along `Try this` from `aesop` -/
macro (name := restrict_tac?) "restrict_tac?" c:Aesop.tactic_clause* : tactic =>
`(tactic|
  aesop? $c*
    (config := { terminal := true
                 assumptionTransparency := .reducible
                 enableSimp := false
                 maxRuleApplications := 300 })
  (rule_sets := [-default, -builtin, $(Lean.mkIdent `Restrict):ident]))


/-- The restriction of a section along an inclusion of open sets.
For `x : F.obj (op V)`, we provide the notation `x |_ₕ i` (`h` stands for `hom`) for `i : U ⟶ V`,
and the notation `x |_ₗ U ⟪i⟫` (`l` stands for `le`) for `i : U ≤ V`.
-/
def restrict {X : TopCat} {C : Type*} [Category C] [ConcreteCategory C] {F : X.Presheaf C}
    {V : Opens X} (x : F.obj (op V)) {U : Opens X} (h : U ⟶ V) : F.obj (op U) :=
  F.map h.op x


/-- restriction of a section along an inclusion -/
scoped[AlgebraicGeometry] infixl:80 " |_ₕ " => TopCat.Presheaf.restrict

/-- restriction of a section along a subset relation -/
scoped[AlgebraicGeometry] notation:80 x " |_ₗ " U " ⟪" e "⟫ " =>
  @TopCat.Presheaf.restrict _ _ _ _ _ _ x U (@homOfLE (Opens _) _ U _ e)


/-- The restriction of a section along an inclusion of open sets.
For `x : F.obj (op V)`, we provide the notation `x |_ U`, where the proof `U ≤ V` is inferred by
the tactic `Top.presheaf.restrict_tac'` -/
abbrev restrictOpen {X : TopCat} {C : Type*} [Category C] [ConcreteCategory C] {F : X.Presheaf C}
    {V : Opens X} (x : F.obj (op V)) (U : Opens X)
    (e : U ≤ V := by restrict_tac) :
    F.obj (op U) :=
  x |_ₗ U ⟪e⟫


/-- restriction of a section to open subset -/
scoped[AlgebraicGeometry] infixl:80 " |_ " => TopCat.Presheaf.restrictOpen

-- Porting note: linter tells this lemma is no going to be picked up by the simplifier, hence
-- `@[simp]` is removed

theorem restrict_restrict {X : TopCat} {C : Type*} [Category C] [ConcreteCategory C]
    {F : X.Presheaf C} {U V W : Opens X} (e₁ : U ≤ V) (e₂ : V ≤ W) (x : F.obj (op W)) :
    x |_ V |_ U = x |_ U := by
  /-
    X : TopCat
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.ConcreteCategory C
    F : TopCat.Presheaf C X
    U V W : TopologicalSpace.Opens ↑X
    e₁ : LE.le U V
    e₂ : LE.le V W
    x : (CategoryTheory.forget C).obj (F.obj { unop := W })
    ⊢ Eq (TopCat.Presheaf.restrictOpen (TopCat.Presheaf.restrictOpen x V e₂) U e₁) …
  -/
  delta restrictOpen restrict
  /-
    X : TopCat
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.ConcreteCategory C
    F : TopCat.Presheaf C X
    U V W : TopologicalSpace.Opens ↑X
    e₁ : LE.le U V
    e₂ : LE.le V W
    x : (CategoryTheory.forget C).obj (F.obj { unop := W })
    ⊢ Eq ((F.map (CategoryTheory.homOfLE e₁).op) ((F.map (CategoryTheory.homOfLE e …
  -/
  rw [← comp_apply, ← Functor.map_comp]
  /-
    X : TopCat
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.ConcreteCategory C
    F : TopCat.Presheaf C X
    U V W : TopologicalSpace.Opens ↑X
    e₁ : LE.le U V
    e₂ : LE.le V W
    x : (CategoryTheory.forget C).obj (F.obj { unop := W })
    ⊢ Eq ((F.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.homOfLE e₂).o …
  -/
  rfl
  /-
    🎉 no goals
  -/

-- Porting note: linter tells this lemma is no going to be picked up by the simplifier, hence
-- `@[simp]` is removed

theorem map_restrict {X : TopCat} {C : Type*} [Category C] [ConcreteCategory C]
    {F G : X.Presheaf C} (e : F ⟶ G) {U V : Opens X} (h : U ≤ V) (x : F.obj (op V)) :
    e.app _ (x |_ U) = e.app _ x |_ U := by
  /-
    X : TopCat
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.ConcreteCategory C
    F G : TopCat.Presheaf C X
    e : Quiver.Hom F G
    U V : TopologicalSpace.Opens ↑X
    h : LE.le U V
    x : (CategoryTheory.forget C).obj (F.obj { unop := V })
    ⊢ Eq ((e.app { unop := U }) (TopCat.Presheaf.restrictOpen x U h)) (TopCat.Pres …
  -/
  delta restrictOpen restrict
  /-
    X : TopCat
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.ConcreteCategory C
    F G : TopCat.Presheaf C X
    e : Quiver.Hom F G
    U V : TopologicalSpace.Opens ↑X
    h : LE.le U V
    x : (CategoryTheory.forget C).obj (F.obj { unop := V })
    ⊢ Eq ((e.app { unop := U }) ((F.map (CategoryTheory.homOfLE h).op) x)) ((G.map …
  -/
  rw [← comp_apply, NatTrans.naturality, comp_apply]
  /-
    🎉 no goals
  -/


/-- The pushforward functor. -/
@[simps!]
def pushforward {X Y : TopCat.{w}} (f : X ⟶ Y) : X.Presheaf C ⥤ Y.Presheaf C :=
  (whiskeringLeft _ _ _).obj (Opens.map f).op


/-- push forward of a presheaf -/
scoped[AlgebraicGeometry] notation f:80 " _* " P:81 =>
  Prefunctor.obj (Functor.toPrefunctor (TopCat.Presheaf.pushforward _ f)) P


@[simp]
theorem pushforward_map_app' {X Y : TopCat.{w}} (f : X ⟶ Y) {ℱ 𝒢 : X.Presheaf C} (α : ℱ ⟶ 𝒢)
    {U : (Opens Y)ᵒᵖ} : ((pushforward C f).map α).app U = α.app (op <| (Opens.map f).obj U.unop) :=
  rfl


lemma id_pushforward (X : TopCat.{w}) : pushforward C (𝟙 X) = 𝟭 (X.Presheaf C) := rfl


/-- The natural isomorphism between the pushforward of a presheaf along the identity continuous map
and the original presheaf. -/
def id {X : TopCat.{w}} (ℱ : X.Presheaf C) : 𝟙 X _* ℱ ≅ ℱ := Iso.refl _


@[simp]
theorem id_hom_app {X : TopCat.{w}} (ℱ : X.Presheaf C) (U) : (id ℱ).hom.app U = 𝟙 _ := rfl


@[simp]
theorem id_inv_app {X : TopCat.{w}} (ℱ : X.Presheaf C) (U) :
    (id ℱ).inv.app U = 𝟙 _ := rfl


theorem id_eq {X : TopCat.{w}} (ℱ : X.Presheaf C) : 𝟙 X _* ℱ = ℱ := rfl


/-- The natural isomorphism between
the pushforward of a presheaf along the composition of two continuous maps and
the corresponding pushforward of a pushforward. -/
def comp {X Y Z : TopCat.{w}} (f : X ⟶ Y) (g : Y ⟶ Z) (ℱ : X.Presheaf C) :
    (f ≫ g) _* ℱ ≅ g _* (f _* ℱ) := Iso.refl _


theorem comp_eq {X Y Z : TopCat.{w}} (f : X ⟶ Y) (g : Y ⟶ Z) (ℱ : X.Presheaf C) :
    (f ≫ g) _* ℱ = g _* (f _* ℱ) :=
  rfl


@[simp]
theorem comp_hom_app {X Y Z : TopCat.{w}} (f : X ⟶ Y) (g : Y ⟶ Z) (ℱ : X.Presheaf C) (U) :
    (comp f g ℱ).hom.app U = 𝟙 _ := rfl


@[simp]
theorem comp_inv_app {X Y Z : TopCat.{w}} (f : X ⟶ Y) (g : Y ⟶ Z) (ℱ : X.Presheaf C) (U) :
    (comp f g ℱ).inv.app U = 𝟙 _ := rfl


/--
An equality of continuous maps induces a natural isomorphism between the pushforwards of a presheaf
along those maps.
-/
def pushforwardEq {X Y : TopCat.{w}} {f g : X ⟶ Y} (h : f = g) (ℱ : X.Presheaf C) :
    f _* ℱ ≅ g _* ℱ :=
  isoWhiskerRight (NatIso.op (Opens.mapIso f g h).symm) ℱ


theorem pushforward_eq' {X Y : TopCat.{w}} {f g : X ⟶ Y} (h : f = g) (ℱ : X.Presheaf C) :
                          /-
                            C : Type u
                            inst✝ : CategoryTheory.Category.{v, u} C
                            X Y : TopCat
                            f g : Quiver.Hom X Y
                            h : Eq f g
                            ℱ : TopCat.Presheaf C X
                            ⊢ Eq ((TopCat.Presheaf.pushforward C f).obj ℱ) ((TopCat.Presheaf.pushforward C …
                          -/
    f _* ℱ = g _* ℱ := by rw [h]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem pushforwardEq_hom_app {X Y : TopCat.{w}} {f g : X ⟶ Y}
    (h : f = g) (ℱ : X.Presheaf C) (U) :
                                                       /-
                                                         C : Type u
                                                         inst✝ : CategoryTheory.Category.{v, u} C
                                                         X Y : TopCat
                                                         f g : Quiver.Hom X Y
                                                         h : Eq f g
                                                         ℱ : TopCat.Presheaf C X
                                                         U : Opposite (TopologicalSpace.Opens ↑Y)
                                                         ⊢ Eq ((TopologicalSpace.Opens.map f).op.obj U) ((TopologicalSpace.Opens.map g) …
                                                       -/
    (pushforwardEq h ℱ).hom.app U = ℱ.map (eqToHom (by aesop_cat)) := by
                                                       /-
                                                         🎉 no goals
                                                       -/
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : TopCat
    f g : Quiver.Hom X Y
    h : Eq f g
    ℱ : TopCat.Presheaf C X
    U : Opposite (TopologicalSpace.Opens ↑Y)
    ⊢ Eq ((TopCat.Presheaf.pushforwardEq h ℱ).hom.app U) (ℱ.map (CategoryTheory.eq …
  -/
  simp [pushforwardEq]
  /-
    🎉 no goals
  -/


/-- A homeomorphism of spaces gives an equivalence of categories of presheaves. -/
@[simps!]
def presheafEquivOfIso {X Y : TopCat} (H : X ≅ Y) : X.Presheaf C ≌ Y.Presheaf C :=
  Equivalence.congrLeft (Opens.mapMapIso H).symm.op


/-- If `H : X ≅ Y` is a homeomorphism,
then given an `H _* ℱ ⟶ 𝒢`, we may obtain an `ℱ ⟶ H ⁻¹ _* 𝒢`.
-/
def toPushforwardOfIso {X Y : TopCat} (H : X ≅ Y) {ℱ : X.Presheaf C} {𝒢 : Y.Presheaf C}
    (α : H.hom _* ℱ ⟶ 𝒢) : ℱ ⟶ H.inv _* 𝒢 :=
  (presheafEquivOfIso _ H).toAdjunction.homEquiv ℱ 𝒢 α


@[simp]
theorem toPushforwardOfIso_app {X Y : TopCat} (H₁ : X ≅ Y) {ℱ : X.Presheaf C} {𝒢 : Y.Presheaf C}
    (H₂ : H₁.hom _* ℱ ⟶ 𝒢) (U : (Opens X)ᵒᵖ) :
    (toPushforwardOfIso H₁ H₂).app U =
                         /-
                           C : Type u
                           inst✝ : CategoryTheory.Category.{v, u} C
                           X Y : TopCat
                           H₁ : CategoryTheory.Iso X Y
                           ℱ : TopCat.Presheaf C X
                           𝒢 : TopCat.Presheaf C Y
                           H₂ : Quiver.Hom ((TopCat.Presheaf.pushforward C H₁.hom).obj ℱ) 𝒢
                           U : Opposite (TopologicalSpace.Opens ↑X)
                           ⊢ Eq U ((TopologicalSpace.Opens.map H₁.hom).op.obj { unop := (TopologicalSpace …
                         -/
      ℱ.map (eqToHom (by simp [Opens.map, Set.preimage_preimage])) ≫
                         /-
                           🎉 no goals
                         -/
        H₂.app (op ((Opens.map H₁.inv).obj (unop U))) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : TopCat
    H₁ : CategoryTheory.Iso X Y
    ℱ : TopCat.Presheaf C X
    𝒢 : TopCat.Presheaf C Y
    H₂ : Quiver.Hom ((TopCat.Presheaf.pushforward C H₁.hom).obj ℱ) 𝒢
    U : Opposite (TopologicalSpace.Opens ↑X)
    ⊢ Eq ((TopCat.Presheaf.toPushforwardOfIso H₁ H₂).app U) (CategoryTheory.Catego …
  -/
  simp [toPushforwardOfIso, Adjunction.homEquiv_unit]
  /-
    🎉 no goals
  -/


/-- If `H : X ≅ Y` is a homeomorphism,
then given an `H _* ℱ ⟶ 𝒢`, we may obtain an `ℱ ⟶ H ⁻¹ _* 𝒢`.
-/
def pushforwardToOfIso {X Y : TopCat} (H₁ : X ≅ Y) {ℱ : Y.Presheaf C} {𝒢 : X.Presheaf C}
    (H₂ : ℱ ⟶ H₁.hom _* 𝒢) : H₁.inv _* ℱ ⟶ 𝒢 :=
  ((presheafEquivOfIso _ H₁.symm).toAdjunction.homEquiv ℱ 𝒢).symm H₂


@[simp]
theorem pushforwardToOfIso_app {X Y : TopCat} (H₁ : X ≅ Y) {ℱ : Y.Presheaf C} {𝒢 : X.Presheaf C}
    (H₂ : ℱ ⟶ H₁.hom _* 𝒢) (U : (Opens X)ᵒᵖ) :
    (pushforwardToOfIso H₁ H₂).app U =
      H₂.app (op ((Opens.map H₁.inv).obj (unop U))) ≫
                           /-
                             C : Type u
                             inst✝ : CategoryTheory.Category.{v, u} C
                             X Y : TopCat
                             H₁ : CategoryTheory.Iso X Y
                             ℱ : TopCat.Presheaf C Y
                             𝒢 : TopCat.Presheaf C X
                             H₂ : Quiver.Hom ℱ ((TopCat.Presheaf.pushforward C H₁.hom).obj 𝒢)
                             U : Opposite (TopologicalSpace.Opens ↑X)
                             ⊢ Eq ((TopologicalSpace.Opens.map H₁.hom).op.obj { unop := (TopologicalSpace.O …
                           -/
        𝒢.map (eqToHom (by simp [Opens.map, Set.preimage_preimage])) := by
                           /-
                             🎉 no goals
                           -/
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : TopCat
    H₁ : CategoryTheory.Iso X Y
    ℱ : TopCat.Presheaf C Y
    𝒢 : TopCat.Presheaf C X
    H₂ : Quiver.Hom ℱ ((TopCat.Presheaf.pushforward C H₁.hom).obj 𝒢)
    U : Opposite (TopologicalSpace.Opens ↑X)
    ⊢ Eq ((TopCat.Presheaf.pushforwardToOfIso H₁ H₂).app U) (CategoryTheory.Catego …
  -/
  simp [pushforwardToOfIso, Equivalence.toAdjunction, Adjunction.homEquiv_counit]
  /-
    🎉 no goals
  -/


/-- Pullback a presheaf on `Y` along a continuous map `f : X ⟶ Y`, obtaining a presheaf
on `X`. -/
def pullback {X Y : TopCat.{v}} (f : X ⟶ Y) : Y.Presheaf C ⥤ X.Presheaf C :=
  (Opens.map f).op.lan


/-- The pullback and pushforward along a continuous map are adjoint to each other. -/
def pushforwardPullbackAdjunction {X Y : TopCat.{v}} (f : X ⟶ Y) :
    pullback C f ⊣ pushforward C f :=
  Functor.lanAdjunction _ _


/-- Pulling back along a homeomorphism is the same as pushing forward along its inverse. -/
def pullbackHomIsoPushforwardInv {X Y : TopCat.{v}} (H : X ≅ Y) :
    pullback C H.hom ≅ pushforward C H.inv :=
  Adjunction.leftAdjointUniq (pushforwardPullbackAdjunction C H.hom)
    (presheafEquivOfIso C H.symm).toAdjunction


/-- Pulling back along the inverse of a homeomorphism is the same as pushing forward along it. -/
def pullbackInvIsoPushforwardHom {X Y : TopCat.{v}} (H : X ≅ Y) :
    pullback C H.inv ≅ pushforward C H.hom :=
  Adjunction.leftAdjointUniq (pushforwardPullbackAdjunction C H.inv)
    (presheafEquivOfIso C H).toAdjunction


/-- If `f '' U` is open, then `f⁻¹ℱ U ≅ ℱ (f '' U)`. -/
def pullbackObjObjOfImageOpen {X Y : TopCat.{v}} (f : X ⟶ Y) (ℱ : Y.Presheaf C) (U : Opens X)
    (H : IsOpen (f '' SetLike.coe U)) : ((pullback C f).obj ℱ).obj (op U) ≅ ℱ.obj (op ⟨_, H⟩) := by
  let x : CostructuredArrow (Opens.map f).op (op U) := CostructuredArrow.mk
    (@homOfLE _ _ _ ((Opens.map f).obj ⟨_, H⟩) (Set.image_preimage.le_u_l _)).op
  have hx : IsTerminal x :=
    { lift := fun s ↦ by
        fapply CostructuredArrow.homMk
        · change op (unop _) ⟶ op (⟨_, H⟩ : Opens _)
          refine (homOfLE ?_).op
          apply (Set.image_subset f s.pt.hom.unop.le).trans
          exact Set.image_preimage.l_u_le (SetLike.coe s.pt.left.unop)
        · simp [eq_iff_true_of_subsingleton] }
  exact IsColimit.coconePointUniqueUpToIso
    ((Opens.map f).op.isPointwiseLeftKanExtensionLeftKanExtensionUnit ℱ (op U))
    (colimitOfDiagramTerminal hx _)


