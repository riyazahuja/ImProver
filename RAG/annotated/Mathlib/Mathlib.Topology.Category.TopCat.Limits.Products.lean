/-- The projection from the product as a bundled continuous map. -/
abbrev piπ {ι : Type v} (α : ι → TopCat.{max v u}) (i : ι) : TopCat.of (∀ i, α i) ⟶ α i :=
  ⟨fun f => f i, continuous_apply i⟩


/-- The explicit fan of a family of topological spaces given by the pi type. -/
@[simps! pt π_app]
def piFan {ι : Type v} (α : ι → TopCat.{max v u}) : Fan α :=
  Fan.mk (TopCat.of (∀ i, α i)) (piπ.{v,u} α)


/-- The constructed fan is indeed a limit -/
def piFanIsLimit {ι : Type v} (α : ι → TopCat.{max v u}) : IsLimit (piFan α) where
  lift S :=
    { toFun := fun s i => S.π.app ⟨i⟩ s
      continuous_toFun := continuous_pi (fun i => (S.π.app ⟨i⟩).2) }
  uniq := by
    /-
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      ι : Type v
      α : ι → TopCat
      ⊢ ∀ (s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor α)) (m :  …
    -/
    intro S m h
    /-
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      ι : Type v
      α : ι → TopCat
      S : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor α)
      m : Quiver.Hom S.pt (TopCat.piFan α).pt
      h : ∀ (j : CategoryTheory.Discrete ι), Eq (CategoryTheory.CategoryStruct.comp  …
      ⊢ Eq m ((fun S => { toFun := fun s i => (S.π.app { as := i }) s, continuous_to …
    -/
    apply ContinuousMap.ext; intro x
    /-
      case h
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      ι : Type v
      α : ι → TopCat
      S : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor α)
      m : Quiver.Hom S.pt (TopCat.piFan α).pt
      h : ∀ (j : CategoryTheory.Discrete ι), Eq (CategoryTheory.CategoryStruct.comp  …
      x : ↑S.pt
      ⊢ Eq (m x) (((fun S => { toFun := fun s i => (S.π.app { as := i }) s, continuo …
    -/
    funext i
    /-
      case h.h
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      ι : Type v
      α : ι → TopCat
      S : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor α)
      m : Quiver.Hom S.pt (TopCat.piFan α).pt
      h : ∀ (j : CategoryTheory.Discrete ι), Eq (CategoryTheory.CategoryStruct.comp  …
      x : ↑S.pt
      i : ι
      ⊢ Eq (m x i) (((fun S => { toFun := fun s i => (S.π.app { as := i }) s, contin …
    -/
    simp [ContinuousMap.coe_mk, ← h ⟨i⟩]
    /-
      case h.h
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      ι : Type v
      α : ι → TopCat
      S : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor α)
      m : Quiver.Hom S.pt (TopCat.piFan α).pt
      h : ∀ (j : CategoryTheory.Discrete ι), Eq (CategoryTheory.CategoryStruct.comp  …
      x : ↑S.pt
      i : ι
      ⊢ Eq (m x i) ((TopCat.piπ α i) (m x))
    -/
    rfl
    /-
      🎉 no goals
    -/
  fac _ _ := rfl


/-- The product is homeomorphic to the product of the underlying spaces,
equipped with the product topology.
-/
def piIsoPi {ι : Type v} (α : ι → TopCat.{max v u}) : ∏ᶜ α ≅ TopCat.of (∀ i, α i) :=
  (limit.isLimit _).conePointUniqueUpToIso (piFanIsLimit.{v, u} α)
  -- Specifying the universes in `piFanIsLimit` wasn't necessary when we had `TopCatMax`


@[reassoc (attr := simp)]
theorem piIsoPi_inv_π {ι : Type v} (α : ι → TopCat.{max v u}) (i : ι) :
                                               /-
                                                 ι : Type v
                                                 α : ι → TopCat
                                                 i : ι
                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (TopCat.piIsoPi α).inv (CategoryTheor …
                                               -/
    (piIsoPi α).inv ≫ Pi.π α i = piπ α i := by simp [piIsoPi]
                                               /-
                                                 🎉 no goals
                                               -/


theorem piIsoPi_inv_π_apply {ι : Type v} (α : ι → TopCat.{max v u}) (i : ι) (x : ∀ i, α i) :
    (Pi.π α i : _) ((piIsoPi α).inv x) = x i :=
  ConcreteCategory.congr_hom (piIsoPi_inv_π α i) x

-- Porting note: needing the type ascription on `∏ᶜ α : TopCat.{max v u}` is unfortunate.

theorem piIsoPi_hom_apply {ι : Type v} (α : ι → TopCat.{max v u}) (i : ι)
    (x : (∏ᶜ α : TopCat.{max v u})) : (piIsoPi α).hom x i = (Pi.π α i : _) x := by
  /-
    ι : Type v
    α : ι → TopCat
    i : ι
    x : ↑(CategoryTheory.Limits.piObj α)
    ⊢ Eq ((TopCat.piIsoPi α).hom x i) ((CategoryTheory.Limits.Pi.π α i) x)
  -/
  have := piIsoPi_inv_π α i
  /-
    ι : Type v
    α : ι → TopCat
    i : ι
    x : ↑(CategoryTheory.Limits.piObj α)
    this : Eq (CategoryTheory.CategoryStruct.comp (TopCat.piIsoPi α).inv (Category …
    ⊢ Eq ((TopCat.piIsoPi α).hom x i) ((CategoryTheory.Limits.Pi.π α i) x)
  -/
  rw [Iso.inv_comp_eq] at this
  /-
    ι : Type v
    α : ι → TopCat
    i : ι
    x : ↑(CategoryTheory.Limits.piObj α)
    this : Eq (CategoryTheory.Limits.Pi.π α i) (CategoryTheory.CategoryStruct.comp …
    ⊢ Eq ((TopCat.piIsoPi α).hom x i) ((CategoryTheory.Limits.Pi.π α i) x)
  -/
  exact ConcreteCategory.congr_hom this x
  /-
    🎉 no goals
  -/

-- Porting note: Lean doesn't automatically reduce TopCat.of X|>.α to X now

/-- The inclusion to the coproduct as a bundled continuous map. -/
abbrev sigmaι {ι : Type v} (α : ι → TopCat.{max v u}) (i : ι) : α i ⟶ TopCat.of (Σi, α i) := by
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    ι : Type v
    α : ι → TopCat
    i : ι
    ⊢ Quiver.Hom (α i) (TopCat.of (Sigma fun i => ↑(α i)))
  -/
  refine ContinuousMap.mk ?_ ?_
    /-
      case refine_1
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      ι : Type v
      α : ι → TopCat
      i : ι
      ⊢ ↑(α i) → ↑(TopCat.of (Sigma fun i => ↑(α i)))
    -/
  · dsimp
    /-
      case refine_1
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      ι : Type v
      α : ι → TopCat
      i : ι
      ⊢ ↑(α i) → Sigma fun i => ↑(α i)
    -/
    apply Sigma.mk i
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      ι : Type v
      α : ι → TopCat
      i : ι
      ⊢ Continuous (id (Sigma.mk i))
    -/
  · dsimp; continuity
           /-
             🎉 no goals
           -/


/-- The explicit cofan of a family of topological spaces given by the sigma type. -/
@[simps! pt ι_app]
def sigmaCofan {ι : Type v} (α : ι → TopCat.{max v u}) : Cofan α :=
  Cofan.mk (TopCat.of (Σi, α i)) (sigmaι α)


/-- The constructed cofan is indeed a colimit -/
def sigmaCofanIsColimit {ι : Type v} (β : ι → TopCat.{max v u}) : IsColimit (sigmaCofan β) where
  desc S :=
    { toFun := fun (s : of (Σ i, β i)) => S.ι.app ⟨s.1⟩ s.2
      continuous_toFun := continuous_sigma fun i => (S.ι.app ⟨i⟩).continuous_toFun }
  uniq := by
    /-
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      ι : Type v
      β : ι → TopCat
      ⊢ ∀ (s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor β)) (m  …
    -/
    intro S m h
    /-
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      ι : Type v
      β : ι → TopCat
      S : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor β)
      m : Quiver.Hom (TopCat.sigmaCofan β).pt S.pt
      h : ∀ (j : CategoryTheory.Discrete ι), Eq (CategoryTheory.CategoryStruct.comp  …
      ⊢ Eq m ((fun S => { toFun := fun s => (S.ι.app { as := s.fst }) s.snd, continu …
    -/
    ext ⟨i, x⟩
    /-
      case w.mk
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      ι : Type v
      β : ι → TopCat
      S : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor β)
      m : Quiver.Hom (TopCat.sigmaCofan β).pt S.pt
      h : ∀ (j : CategoryTheory.Discrete ι), Eq (CategoryTheory.CategoryStruct.comp  …
      i : ι
      x : ↑(β i)
      ⊢ Eq (m ⟨i, x⟩) (((fun S => { toFun := fun s => (S.ι.app { as := s.fst }) s.sn …
    -/
    /-
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      ι : Type v
      β : ι → TopCat
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor β)
      j : CategoryTheory.Discrete ι
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((TopCat.sigmaCofan β).ι.app j) ((fun …
    -/
    simp only [hom_apply, ← h]
    /-
      case mk
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      ι : Type v
      β : ι → TopCat
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor β)
      as✝ : ι
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((TopCat.sigmaCofan β).ι.app { as :=  …
    -/
    /-
      case w.mk
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      ι : Type v
      β : ι → TopCat
      S : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor β)
      m : Quiver.Hom (TopCat.sigmaCofan β).pt S.pt
      h : ∀ (j : CategoryTheory.Discrete ι), Eq (CategoryTheory.CategoryStruct.comp  …
      i : ι
      x : ↑(β i)
      ⊢ Eq (m ⟨i, x⟩) ({ toFun := fun s => (CategoryTheory.CategoryStruct.comp ((Top …
    -/
    /-
      🎉 no goals
    -/
    congr
    /-
      🎉 no goals
    -/
  fac s j := by
    cases j
    aesop_cat


/-- The coproduct is homeomorphic to the disjoint union of the topological spaces.
-/
def sigmaIsoSigma {ι : Type v} (α : ι → TopCat.{max v u}) : ∐ α ≅ TopCat.of (Σi, α i) :=
  (colimit.isColimit _).coconePointUniqueUpToIso (sigmaCofanIsColimit.{v, u} α)
  -- Specifying the universes in `sigmaCofanIsColimit` wasn't necessary when we had `TopCatMax`


@[reassoc (attr := simp)]
theorem sigmaIsoSigma_hom_ι {ι : Type v} (α : ι → TopCat.{max v u}) (i : ι) :
                                                           /-
                                                             ι : Type v
                                                             α : ι → TopCat
                                                             i : ι
                                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι α i) ( …
                                                           -/
    Sigma.ι α i ≫ (sigmaIsoSigma α).hom = sigmaι α i := by simp [sigmaIsoSigma]
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem sigmaIsoSigma_hom_ι_apply {ι : Type v} (α : ι → TopCat.{max v u}) (i : ι) (x : α i) :
    (sigmaIsoSigma α).hom ((Sigma.ι α i : _) x) = Sigma.mk i x :=
  ConcreteCategory.congr_hom (sigmaIsoSigma_hom_ι α i) x


theorem sigmaIsoSigma_inv_apply {ι : Type v} (α : ι → TopCat.{max v u}) (i : ι) (x : α i) :
    (sigmaIsoSigma α).inv ⟨i, x⟩ = (Sigma.ι α i : _) x := by
  rw [← sigmaIsoSigma_hom_ι_apply, ← comp_app, ← comp_app, Iso.hom_inv_id,
    Category.comp_id]

-- Porting note: cannot use .topologicalSpace in place .str

theorem induced_of_isLimit {F : J ⥤ TopCat.{max v u}} (C : Cone F) (hC : IsLimit C) :
    C.pt.str = ⨅ j, (F.obj j).str.induced (C.π.app j) := by
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J TopCat
    C : CategoryTheory.Limits.Cone F
    hC : CategoryTheory.Limits.IsLimit C
    ⊢ Eq C.pt.str (iInf fun j => TopologicalSpace.induced (⇑(C.π.app j)) (F.obj j) …
  -/
  let homeo := homeoOfIso (hC.conePointUniqueUpToIso (limitConeInfiIsLimit F))
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J TopCat
    C : CategoryTheory.Limits.Cone F
    hC : CategoryTheory.Limits.IsLimit C
    homeo : Homeomorph ↑C.pt ↑(TopCat.limitConeInfi F).pt := TopCat.homeoOfIso (hC …
    ⊢ Eq C.pt.str (iInf fun j => TopologicalSpace.induced (⇑(C.π.app j)) (F.obj j) …
  -/
  refine homeo.isInducing.eq_induced.trans ?_
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J TopCat
    C : CategoryTheory.Limits.Cone F
    hC : CategoryTheory.Limits.IsLimit C
    homeo : Homeomorph ↑C.pt ↑(TopCat.limitConeInfi F).pt := TopCat.homeoOfIso (hC …
    ⊢ Eq (TopologicalSpace.induced (⇑homeo) (TopCat.limitConeInfi F).pt.topologica …
  -/
  change induced homeo (⨅ j : J, _) = _
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J TopCat
    C : CategoryTheory.Limits.Cone F
    hC : CategoryTheory.Limits.IsLimit C
    homeo : Homeomorph ↑C.pt ↑(TopCat.limitConeInfi F).pt := TopCat.homeoOfIso (hC …
    ⊢ Eq (TopologicalSpace.induced (⇑homeo) (iInf fun j => TopologicalSpace.induce …
  -/
  simp [induced_iInf, induced_compose]
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J TopCat
    C : CategoryTheory.Limits.Cone F
    hC : CategoryTheory.Limits.IsLimit C
    homeo : Homeomorph ↑C.pt ↑(TopCat.limitConeInfi F).pt := TopCat.homeoOfIso (hC …
    ⊢ Eq (iInf fun i => TopologicalSpace.induced (Function.comp ((CategoryTheory.L …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem limit_topology (F : J ⥤ TopCat.{max v u}) :
    (limit F).str = ⨅ j, (F.obj j).str.induced (limit.π F j) :=
  induced_of_isLimit _ (limit.isLimit F)


/-- The first projection from the product. -/
abbrev prodFst {X Y : TopCat.{u}} : TopCat.of (X × Y) ⟶ X :=
                /-
                  J : Type v
                  inst✝ : CategoryTheory.Category.{w, v} J
                  X Y : TopCat
                  ⊢ Continuous Prod.fst
                -/
  ⟨Prod.fst, by continuity⟩
                /-
                  🎉 no goals
                -/


/-- The second projection from the product. -/
abbrev prodSnd {X Y : TopCat.{u}} : TopCat.of (X × Y) ⟶ Y :=
                /-
                  J : Type v
                  inst✝ : CategoryTheory.Category.{w, v} J
                  X Y : TopCat
                  ⊢ Continuous Prod.snd
                -/
  ⟨Prod.snd, by continuity⟩
                /-
                  🎉 no goals
                -/


/-- The explicit binary cofan of `X, Y` given by `X × Y`. -/
def prodBinaryFan (X Y : TopCat.{u}) : BinaryFan X Y :=
  BinaryFan.mk prodFst prodSnd


/-- The constructed binary fan is indeed a limit -/
def prodBinaryFanIsLimit (X Y : TopCat.{u}) : IsLimit (prodBinaryFan X Y) where
  lift := fun S : BinaryFan X Y => {
    toFun := fun s => (S.fst s, S.snd s)
    -- Porting note: continuity failed again here. Lean cannot infer
    -- ContinuousMapClass (X ⟶ Y) X Y for X Y : TopCat which may be one of the problems
    continuous_toFun := Continuous.prod_mk
      (BinaryFan.fst S).continuous_toFun (BinaryFan.snd S).continuous_toFun }
  fac := by
    /-
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      X Y : TopCat
      ⊢ ∀ (s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X Y)) (j : Cat …
    -/
                         /-
                           🎉 no goals
                         -/
    rintro S (_ | _) <;> {dsimp; ext; rfl}
                         /-
                           🎉 no goals
                         -/
  uniq := by
    /-
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      X Y : TopCat
      ⊢ ∀ (s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X Y)) (m : Qui …
    -/
    intro S m h
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): used to be `ext x`
    /-
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      X Y : TopCat
      S : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X Y)
      m : Quiver.Hom S.pt (X.prodBinaryFan Y).pt
      h : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
      ⊢ Eq m ((fun S => { toFun := fun s => { fst := S.fst s, snd := S.snd s }, cont …
    -/
    refine ContinuousMap.ext (fun (x : ↥(S.pt)) => Prod.ext ?_ ?_)
      /-
        case refine_1
        J : Type v
        inst✝ : CategoryTheory.Category.{w, v} J
        X Y : TopCat
        S : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X Y)
        m : Quiver.Hom S.pt (X.prodBinaryFan Y).pt
        h : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
        x : ↑S.pt
        ⊢ Eq (m x).1 (((fun S => { toFun := fun s => { fst := S.fst s, snd := S.snd s  …
      -/
    · specialize h ⟨WalkingPair.left⟩
      /-
        case refine_1
        J : Type v
        inst✝ : CategoryTheory.Category.{w, v} J
        X Y : TopCat
        S : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X Y)
        m : Quiver.Hom S.pt (X.prodBinaryFan Y).pt
        x : ↑S.pt
        h : Eq (CategoryTheory.CategoryStruct.comp m ((X.prodBinaryFan Y).π.app { as : …
        ⊢ Eq (m x).1 (((fun S => { toFun := fun s => { fst := S.fst s, snd := S.snd s  …
      -/
      apply_fun fun e => e x at h
      /-
        case refine_1
        J : Type v
        inst✝ : CategoryTheory.Category.{w, v} J
        X Y : TopCat
        S : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X Y)
        m : Quiver.Hom S.pt (X.prodBinaryFan Y).pt
        x : ↑S.pt
        h : Eq ((CategoryTheory.CategoryStruct.comp m ((X.prodBinaryFan Y).π.app { as  …
        ⊢ Eq (m x).1 (((fun S => { toFun := fun s => { fst := S.fst s, snd := S.snd s  …
      -/
      exact h
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        J : Type v
        inst✝ : CategoryTheory.Category.{w, v} J
        X Y : TopCat
        S : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X Y)
        m : Quiver.Hom S.pt (X.prodBinaryFan Y).pt
        h : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
        x : ↑S.pt
        ⊢ Eq (m x).2 (((fun S => { toFun := fun s => { fst := S.fst s, snd := S.snd s  …
      -/
    · specialize h ⟨WalkingPair.right⟩
      /-
        case refine_2
        J : Type v
        inst✝ : CategoryTheory.Category.{w, v} J
        X Y : TopCat
        S : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X Y)
        m : Quiver.Hom S.pt (X.prodBinaryFan Y).pt
        x : ↑S.pt
        h : Eq (CategoryTheory.CategoryStruct.comp m ((X.prodBinaryFan Y).π.app { as : …
        ⊢ Eq (m x).2 (((fun S => { toFun := fun s => { fst := S.fst s, snd := S.snd s  …
      -/
      apply_fun fun e => e x at h
      /-
        case refine_2
        J : Type v
        inst✝ : CategoryTheory.Category.{w, v} J
        X Y : TopCat
        S : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X Y)
        m : Quiver.Hom S.pt (X.prodBinaryFan Y).pt
        x : ↑S.pt
        h : Eq ((CategoryTheory.CategoryStruct.comp m ((X.prodBinaryFan Y).π.app { as  …
        ⊢ Eq (m x).2 (((fun S => { toFun := fun s => { fst := S.fst s, snd := S.snd s  …
      -/
      exact h
      /-
        🎉 no goals
      -/


/-- The homeomorphism between `X ⨯ Y` and the set-theoretic product of `X` and `Y`,
equipped with the product topology.
-/
def prodIsoProd (X Y : TopCat.{u}) : X ⨯ Y ≅ TopCat.of (X × Y) :=
  (limit.isLimit _).conePointUniqueUpToIso (prodBinaryFanIsLimit X Y)


@[reassoc (attr := simp)]
theorem prodIsoProd_hom_fst (X Y : TopCat.{u}) :
    (prodIsoProd X Y).hom ≫ prodFst = Limits.prod.fst := by
  /-
    X Y : TopCat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.prodIsoProd Y).hom TopCat.prodFst) …
  -/
  simp [← Iso.eq_inv_comp, prodIsoProd]
  /-
    X Y : TopCat
    ⊢ Eq TopCat.prodFst (X.prodBinaryFan Y).fst
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem prodIsoProd_hom_snd (X Y : TopCat.{u}) :
    (prodIsoProd X Y).hom ≫ prodSnd = Limits.prod.snd := by
  /-
    X Y : TopCat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.prodIsoProd Y).hom TopCat.prodSnd) …
  -/
  simp [← Iso.eq_inv_comp, prodIsoProd]
  /-
    X Y : TopCat
    ⊢ Eq TopCat.prodSnd (X.prodBinaryFan Y).snd
  -/
  rfl
  /-
    🎉 no goals
  -/

-- Porting note: need to force Lean to coerce X × Y to a type

theorem prodIsoProd_hom_apply {X Y : TopCat.{u}} (x : ↑ (X ⨯ Y)) :
    (prodIsoProd X Y).hom x = ((Limits.prod.fst : X ⨯ Y ⟶ _) x,
    (Limits.prod.snd : X ⨯ Y ⟶ _) x) := by
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` didn't pick this up.
  /-
    X Y : TopCat
    x : ↑(CategoryTheory.Limits.prod X Y)
    ⊢ Eq ((X.prodIsoProd Y).hom x) { fst := CategoryTheory.Limits.prod.fst x, snd  …
  -/
  apply Prod.ext
    /-
      case fst
      X Y : TopCat
      x : ↑(CategoryTheory.Limits.prod X Y)
      ⊢ Eq ((X.prodIsoProd Y).hom x).1 { fst := CategoryTheory.Limits.prod.fst x, sn …
    -/
  · exact ConcreteCategory.congr_hom (prodIsoProd_hom_fst X Y) x
    /-
      🎉 no goals
    -/
    /-
      case snd
      X Y : TopCat
      x : ↑(CategoryTheory.Limits.prod X Y)
      ⊢ Eq ((X.prodIsoProd Y).hom x).2 { fst := CategoryTheory.Limits.prod.fst x, sn …
    -/
  · exact ConcreteCategory.congr_hom (prodIsoProd_hom_snd X Y) x
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp), elementwise]
theorem prodIsoProd_inv_fst (X Y : TopCat.{u}) :
                                                            /-
                                                              X Y : TopCat
                                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.prodIsoProd Y).inv CategoryTheory. …
                                                            -/
    (prodIsoProd X Y).inv ≫ Limits.prod.fst = prodFst := by simp [Iso.inv_comp_eq]
                                                            /-
                                                              🎉 no goals
                                                            -/


@[reassoc (attr := simp), elementwise]
theorem prodIsoProd_inv_snd (X Y : TopCat.{u}) :
                                                            /-
                                                              X Y : TopCat
                                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.prodIsoProd Y).inv CategoryTheory. …
                                                            -/
    (prodIsoProd X Y).inv ≫ Limits.prod.snd = prodSnd := by simp [Iso.inv_comp_eq]
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem prod_topology {X Y : TopCat.{u}} :
    (X ⨯ Y).str =
      induced (Limits.prod.fst : X ⨯ Y ⟶ _) X.str ⊓
        induced (Limits.prod.snd : X ⨯ Y ⟶ _) Y.str := by
  /-
    X Y : TopCat
    ⊢ Eq (CategoryTheory.Limits.prod X Y).str (Min.min (TopologicalSpace.induced ( …
  -/
  let homeo := homeoOfIso (prodIsoProd X Y)
  /-
    X Y : TopCat
    homeo : Homeomorph ↑(CategoryTheory.Limits.prod X Y) ↑(TopCat.of (Prod ↑X ↑Y)) …
    ⊢ Eq (CategoryTheory.Limits.prod X Y).str (Min.min (TopologicalSpace.induced ( …
  -/
  refine homeo.isInducing.eq_induced.trans ?_
  /-
    X Y : TopCat
    homeo : Homeomorph ↑(CategoryTheory.Limits.prod X Y) ↑(TopCat.of (Prod ↑X ↑Y)) …
    ⊢ Eq (TopologicalSpace.induced (⇑homeo) (TopCat.of (Prod ↑X ↑Y)).topologicalSp …
  -/
  change induced homeo (_ ⊓ _) = _
  /-
    X Y : TopCat
    homeo : Homeomorph ↑(CategoryTheory.Limits.prod X Y) ↑(TopCat.of (Prod ↑X ↑Y)) …
    ⊢ Eq (TopologicalSpace.induced (⇑homeo) (Min.min (TopologicalSpace.induced Pro …
  -/
  simp [induced_compose]
  /-
    X Y : TopCat
    homeo : Homeomorph ↑(CategoryTheory.Limits.prod X Y) ↑(TopCat.of (Prod ↑X ↑Y)) …
    ⊢ Eq (Min.min (TopologicalSpace.induced (Function.comp Prod.fst ⇑homeo) X.topo …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem range_prod_map {W X Y Z : TopCat.{u}} (f : W ⟶ Y) (g : X ⟶ Z) :
    Set.range (Limits.prod.map f g) =
      (Limits.prod.fst : Y ⨯ Z ⟶ _) ⁻¹' Set.range f ∩
        (Limits.prod.snd : Y ⨯ Z ⟶ _) ⁻¹' Set.range g := by
  /-
    W X Y Z : TopCat
    f : Quiver.Hom W Y
    g : Quiver.Hom X Z
    ⊢ Eq (Set.range ⇑(CategoryTheory.Limits.prod.map f g)) (Inter.inter (Set.preim …
  -/
  ext x
  /-
    case h
    W X Y Z : TopCat
    f : Quiver.Hom W Y
    g : Quiver.Hom X Z
    x : ↑(CategoryTheory.Limits.prod Y Z)
    ⊢ Iff (Membership.mem (Set.range ⇑(CategoryTheory.Limits.prod.map f g)) x) (Me …
  -/
  constructor
    /-
      case h.mp
      W X Y Z : TopCat
      f : Quiver.Hom W Y
      g : Quiver.Hom X Z
      x : ↑(CategoryTheory.Limits.prod Y Z)
      ⊢ Membership.mem (Set.range ⇑(CategoryTheory.Limits.prod.map f g)) x → Members …
    -/
  · rintro ⟨y, rfl⟩
    /-
      case h.mp.intro
      W X Y Z : TopCat
      f : Quiver.Hom W Y
      g : Quiver.Hom X Z
      y : ↑(CategoryTheory.Limits.prod W X)
      ⊢ Membership.mem (Inter.inter (Set.preimage (⇑CategoryTheory.Limits.prod.fst)  …
    -/
    simp_rw [Set.mem_inter_iff, Set.mem_preimage, Set.mem_range]
    -- sizable changes in this proof after https://github.com/leanprover-community/mathlib4/pull/13170
    /-
      case h.mp.intro
      W X Y Z : TopCat
      f : Quiver.Hom W Y
      g : Quiver.Hom X Z
      y : ↑(CategoryTheory.Limits.prod W X)
      ⊢ And (Exists fun y_1 => Eq (f y_1) (CategoryTheory.Limits.prod.fst ((Category …
    -/
    rw [← comp_apply, ← comp_apply]
    simp_rw [Limits.prod.map_fst,
      Limits.prod.map_snd, comp_apply]
    /-
      case h.mp.intro
      W X Y Z : TopCat
      f : Quiver.Hom W Y
      g : Quiver.Hom X Z
      y : ↑(CategoryTheory.Limits.prod W X)
      ⊢ And (Exists fun y_1 => Eq (f y_1) (f (CategoryTheory.Limits.prod.fst y))) (E …
    -/
    exact ⟨exists_apply_eq_apply _ _, exists_apply_eq_apply _ _⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      W X Y Z : TopCat
      f : Quiver.Hom W Y
      g : Quiver.Hom X Z
      x : ↑(CategoryTheory.Limits.prod Y Z)
      ⊢ Membership.mem (Inter.inter (Set.preimage (⇑CategoryTheory.Limits.prod.fst)  …
    -/
  · rintro ⟨⟨x₁, hx₁⟩, ⟨x₂, hx₂⟩⟩
    /-
      case h.mpr.intro.intro.intro
      W X Y Z : TopCat
      f : Quiver.Hom W Y
      g : Quiver.Hom X Z
      x : ↑(CategoryTheory.Limits.prod Y Z)
      x₁ : ↑W
      hx₁ : Eq (f x₁) (CategoryTheory.Limits.prod.fst x)
      x₂ : ↑X
      hx₂ : Eq (g x₂) (CategoryTheory.Limits.prod.snd x)
      ⊢ Membership.mem (Set.range ⇑(CategoryTheory.Limits.prod.map f g)) x
    -/
    use (prodIsoProd W X).inv (x₁, x₂)
    /-
      case h
      W X Y Z : TopCat
      f : Quiver.Hom W Y
      g : Quiver.Hom X Z
      x : ↑(CategoryTheory.Limits.prod Y Z)
      x₁ : ↑W
      hx₁ : Eq (f x₁) (CategoryTheory.Limits.prod.fst x)
      x₂ : ↑X
      hx₂ : Eq (g x₂) (CategoryTheory.Limits.prod.snd x)
      ⊢ Eq ((CategoryTheory.Limits.prod.map f g) ((W.prodIsoProd X).inv { fst := x₁, …
    -/
    change (forget TopCat).map _ _ = _
    /-
      case h
      W X Y Z : TopCat
      f : Quiver.Hom W Y
      g : Quiver.Hom X Z
      x : ↑(CategoryTheory.Limits.prod Y Z)
      x₁ : ↑W
      hx₁ : Eq (f x₁) (CategoryTheory.Limits.prod.fst x)
      x₂ : ↑X
      hx₂ : Eq (g x₂) (CategoryTheory.Limits.prod.snd x)
      ⊢ Eq ((CategoryTheory.forget TopCat).map (CategoryTheory.Limits.prod.map f g)  …
    -/
    apply Concrete.limit_ext
    /-
      case h.a
      W X Y Z : TopCat
      f : Quiver.Hom W Y
      g : Quiver.Hom X Z
      x : ↑(CategoryTheory.Limits.prod Y Z)
      x₁ : ↑W
      hx₁ : Eq (f x₁) (CategoryTheory.Limits.prod.fst x)
      x₂ : ↑X
      hx₂ : Eq (g x₂) (CategoryTheory.Limits.prod.snd x)
      ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq ((Cate …
    -/
    rintro ⟨⟨⟩⟩
      /-
        case h.a.mk.left
        W X Y Z : TopCat
        f : Quiver.Hom W Y
        g : Quiver.Hom X Z
        x : ↑(CategoryTheory.Limits.prod Y Z)
        x₁ : ↑W
        hx₁ : Eq (f x₁) (CategoryTheory.Limits.prod.fst x)
        x₂ : ↑X
        hx₂ : Eq (g x₂) (CategoryTheory.Limits.prod.snd x)
        ⊢ Eq ((CategoryTheory.Limits.limit.π (CategoryTheory.Limits.pair Y Z) { as :=  …
      -/
    · change limit.π (pair Y Z) _ ((prod.map f g) _) = _
      /-
        case h.a.mk.left
        W X Y Z : TopCat
        f : Quiver.Hom W Y
        g : Quiver.Hom X Z
        x : ↑(CategoryTheory.Limits.prod Y Z)
        x₁ : ↑W
        hx₁ : Eq (f x₁) (CategoryTheory.Limits.prod.fst x)
        x₂ : ↑X
        hx₂ : Eq (g x₂) (CategoryTheory.Limits.prod.snd x)
        ⊢ Eq ((CategoryTheory.Limits.limit.π (CategoryTheory.Limits.pair Y Z) { as :=  …
      -/
      erw [← comp_apply, Limits.prod.map_fst]
      /-
        case h.a.mk.left
        W X Y Z : TopCat
        f : Quiver.Hom W Y
        g : Quiver.Hom X Z
        x : ↑(CategoryTheory.Limits.prod Y Z)
        x₁ : ↑W
        hx₁ : Eq (f x₁) (CategoryTheory.Limits.prod.fst x)
        x₂ : ↑X
        hx₂ : Eq (g x₂) (CategoryTheory.Limits.prod.snd x)
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.prod.fst f) (( …
      -/
      change (_ ≫ _ ≫ f) _ = _
      /-
        case h.a.mk.left
        W X Y Z : TopCat
        f : Quiver.Hom W Y
        g : Quiver.Hom X Z
        x : ↑(CategoryTheory.Limits.prod Y Z)
        x₁ : ↑W
        hx₁ : Eq (f x₁) (CategoryTheory.Limits.prod.fst x)
        x₂ : ↑X
        hx₂ : Eq (g x₂) (CategoryTheory.Limits.prod.snd x)
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (W.prodIsoProd X).inv (CategoryTheor …
      -/
      rw [TopCat.prodIsoProd_inv_fst_assoc,TopCat.comp_app]
      /-
        case h.a.mk.left
        W X Y Z : TopCat
        f : Quiver.Hom W Y
        g : Quiver.Hom X Z
        x : ↑(CategoryTheory.Limits.prod Y Z)
        x₁ : ↑W
        hx₁ : Eq (f x₁) (CategoryTheory.Limits.prod.fst x)
        x₂ : ↑X
        hx₂ : Eq (g x₂) (CategoryTheory.Limits.prod.snd x)
        ⊢ Eq (f (TopCat.prodFst { fst := x₁, snd := x₂ })) ((CategoryTheory.Limits.lim …
      -/
      exact hx₁
      /-
        🎉 no goals
      -/
      /-
        case h.a.mk.right
        W X Y Z : TopCat
        f : Quiver.Hom W Y
        g : Quiver.Hom X Z
        x : ↑(CategoryTheory.Limits.prod Y Z)
        x₁ : ↑W
        hx₁ : Eq (f x₁) (CategoryTheory.Limits.prod.fst x)
        x₂ : ↑X
        hx₂ : Eq (g x₂) (CategoryTheory.Limits.prod.snd x)
        ⊢ Eq ((CategoryTheory.Limits.limit.π (CategoryTheory.Limits.pair Y Z) { as :=  …
      -/
    · change limit.π (pair Y Z) _ ((prod.map f g) _) = _
      /-
        case h.a.mk.right
        W X Y Z : TopCat
        f : Quiver.Hom W Y
        g : Quiver.Hom X Z
        x : ↑(CategoryTheory.Limits.prod Y Z)
        x₁ : ↑W
        hx₁ : Eq (f x₁) (CategoryTheory.Limits.prod.fst x)
        x₂ : ↑X
        hx₂ : Eq (g x₂) (CategoryTheory.Limits.prod.snd x)
        ⊢ Eq ((CategoryTheory.Limits.limit.π (CategoryTheory.Limits.pair Y Z) { as :=  …
      -/
      erw [← comp_apply, Limits.prod.map_snd]
      /-
        case h.a.mk.right
        W X Y Z : TopCat
        f : Quiver.Hom W Y
        g : Quiver.Hom X Z
        x : ↑(CategoryTheory.Limits.prod Y Z)
        x₁ : ↑W
        hx₁ : Eq (f x₁) (CategoryTheory.Limits.prod.fst x)
        x₂ : ↑X
        hx₂ : Eq (g x₂) (CategoryTheory.Limits.prod.snd x)
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.prod.snd g) (( …
      -/
      change (_ ≫ _ ≫ g) _ = _
      /-
        case h.a.mk.right
        W X Y Z : TopCat
        f : Quiver.Hom W Y
        g : Quiver.Hom X Z
        x : ↑(CategoryTheory.Limits.prod Y Z)
        x₁ : ↑W
        hx₁ : Eq (f x₁) (CategoryTheory.Limits.prod.fst x)
        x₂ : ↑X
        hx₂ : Eq (g x₂) (CategoryTheory.Limits.prod.snd x)
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (W.prodIsoProd X).inv (CategoryTheor …
      -/
      rw [TopCat.prodIsoProd_inv_snd_assoc,TopCat.comp_app]
      /-
        case h.a.mk.right
        W X Y Z : TopCat
        f : Quiver.Hom W Y
        g : Quiver.Hom X Z
        x : ↑(CategoryTheory.Limits.prod Y Z)
        x₁ : ↑W
        hx₁ : Eq (f x₁) (CategoryTheory.Limits.prod.fst x)
        x₂ : ↑X
        hx₂ : Eq (g x₂) (CategoryTheory.Limits.prod.snd x)
        ⊢ Eq (g (TopCat.prodSnd { fst := x₁, snd := x₂ })) ((CategoryTheory.Limits.lim …
      -/
      exact hx₂
      /-
        🎉 no goals
      -/


theorem isInducing_prodMap {W X Y Z : TopCat.{u}} {f : W ⟶ X} {g : Y ⟶ Z} (hf : IsInducing f)
    (hg : IsInducing g) : IsInducing (Limits.prod.map f g) := by
  /-
    W X Y Z : TopCat
    f : Quiver.Hom W X
    g : Quiver.Hom Y Z
    hf : Topology.IsInducing ⇑f
    hg : Topology.IsInducing ⇑g
    ⊢ Topology.IsInducing ⇑(CategoryTheory.Limits.prod.map f g)
  -/
  constructor
  simp_rw [topologicalSpace_coe, prod_topology, induced_inf, induced_compose, ← coe_comp,
    prod.map_fst, prod.map_snd, coe_comp, ← induced_compose (g := f), ← induced_compose (g := g)]
  /-
    case eq_induced
    W X Y Z : TopCat
    f : Quiver.Hom W X
    g : Quiver.Hom Y Z
    hf : Topology.IsInducing ⇑f
    hg : Topology.IsInducing ⇑g
    ⊢ Eq (Min.min (TopologicalSpace.induced (⇑CategoryTheory.Limits.prod.fst) W.st …
  -/
  erw [← hf.eq_induced, ← hg.eq_induced] -- now `erw` after https://github.com/leanprover-community/mathlib4/pull/13170
  /-
    case eq_induced
    W X Y Z : TopCat
    f : Quiver.Hom W X
    g : Quiver.Hom Y Z
    hf : Topology.IsInducing ⇑f
    hg : Topology.IsInducing ⇑g
    ⊢ Eq (Min.min (TopologicalSpace.induced (⇑CategoryTheory.Limits.prod.fst) W.st …
  -/
  rfl -- `rfl` was not needed before https://github.com/leanprover-community/mathlib4/pull/13170
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-28")] alias inducing_prod_map := isInducing_prodMap


theorem isEmbedding_prodMap {W X Y Z : TopCat.{u}} {f : W ⟶ X} {g : Y ⟶ Z} (hf : IsEmbedding f)
    (hg : IsEmbedding g) : IsEmbedding (Limits.prod.map f g) :=
  ⟨isInducing_prodMap hf.isInducing hg.isInducing, by
    /-
      W X Y Z : TopCat
      f : Quiver.Hom W X
      g : Quiver.Hom Y Z
      hf : Topology.IsEmbedding ⇑f
      hg : Topology.IsEmbedding ⇑g
      ⊢ Function.Injective ⇑(CategoryTheory.Limits.prod.map f g)
    -/
    haveI := (TopCat.mono_iff_injective _).mpr hf.injective
    /-
      W X Y Z : TopCat
      f : Quiver.Hom W X
      g : Quiver.Hom Y Z
      hf : Topology.IsEmbedding ⇑f
      hg : Topology.IsEmbedding ⇑g
      this : CategoryTheory.Mono f
      ⊢ Function.Injective ⇑(CategoryTheory.Limits.prod.map f g)
    -/
    haveI := (TopCat.mono_iff_injective _).mpr hg.injective
    /-
      W X Y Z : TopCat
      f : Quiver.Hom W X
      g : Quiver.Hom Y Z
      hf : Topology.IsEmbedding ⇑f
      hg : Topology.IsEmbedding ⇑g
      this✝ : CategoryTheory.Mono f
      this : CategoryTheory.Mono g
      ⊢ Function.Injective ⇑(CategoryTheory.Limits.prod.map f g)
    -/
    exact (TopCat.mono_iff_injective _).mp inferInstance⟩
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-26")]
alias embedding_prod_map := isEmbedding_prodMap


/-- The binary coproduct cofan in `TopCat`. -/
protected def binaryCofan (X Y : TopCat.{u}) : BinaryCofan X Y :=
                               /-
                                 J : Type v
                                 inst✝ : CategoryTheory.Category.{w, v} J
                                 X Y : TopCat
                                 ⊢ Continuous Sum.inl
                               -/
                               /-
                                 🎉 no goals
                               -/
  BinaryCofan.mk (⟨Sum.inl, by continuity⟩ : X ⟶ TopCat.of (X ⊕ Y)) ⟨Sum.inr, by continuity⟩
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


/-- The constructed binary coproduct cofan in `TopCat` is the coproduct. -/
def binaryCofanIsColimit (X Y : TopCat.{u}) : IsColimit (TopCat.binaryCofan X Y) := by
  refine Limits.BinaryCofan.isColimitMk (fun s =>
    {toFun := Sum.elim s.inl s.inr, continuous_toFun := ?_ }) ?_ ?_ ?_
  · apply
      Continuous.sum_elim (BinaryCofan.inl s).continuous_toFun (BinaryCofan.inr s).continuous_toFun
    /-
      case refine_2
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      X Y : TopCat
      ⊢ ∀ (s : CategoryTheory.Limits.BinaryCofan X Y), Eq (CategoryTheory.CategorySt …
    -/
  · intro s
    /-
      case refine_2
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      X Y : TopCat
      s : CategoryTheory.Limits.BinaryCofan X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := Sum.inl, continuous_toFun  …
    -/
    ext
    /-
      case refine_2.w
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      X Y : TopCat
      s : CategoryTheory.Limits.BinaryCofan X Y
      x✝ : (CategoryTheory.forget TopCat).obj X
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp { toFun := Sum.inl, continuous_toFun …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      X Y : TopCat
      ⊢ ∀ (s : CategoryTheory.Limits.BinaryCofan X Y), Eq (CategoryTheory.CategorySt …
    -/
  · intro s
    /-
      case refine_3
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      X Y : TopCat
      s : CategoryTheory.Limits.BinaryCofan X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := Sum.inr, continuous_toFun  …
    -/
    ext
    /-
      case refine_3.w
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      X Y : TopCat
      s : CategoryTheory.Limits.BinaryCofan X Y
      x✝ : (CategoryTheory.forget TopCat).obj Y
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp { toFun := Sum.inr, continuous_toFun …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      X Y : TopCat
      ⊢ ∀ (s : CategoryTheory.Limits.BinaryCofan X Y) (m : Quiver.Hom (TopCat.of (Su …
    -/
  · intro s m h₁ h₂
    /-
      case refine_4
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      X Y : TopCat
      s : CategoryTheory.Limits.BinaryCofan X Y
      m : Quiver.Hom (TopCat.of (Sum ↑X ↑Y)) s.pt
      h₁ : Eq (CategoryTheory.CategoryStruct.comp { toFun := Sum.inl, continuous_toF …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp { toFun := Sum.inr, continuous_toF …
      ⊢ Eq m ((fun s => { toFun := Sum.elim ⇑s.inl ⇑s.inr, continuous_toFun := ⋯ }) s)
    -/
    ext (x | x)
    /-
      case refine_4.w.inl
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      X Y : TopCat
      s : CategoryTheory.Limits.BinaryCofan X Y
      m : Quiver.Hom (TopCat.of (Sum ↑X ↑Y)) s.pt
      h₁ : Eq (CategoryTheory.CategoryStruct.comp { toFun := Sum.inl, continuous_toF …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp { toFun := Sum.inr, continuous_toF …
      x : ↑X
      ⊢ Eq (m (Sum.inl x)) (((fun s => { toFun := Sum.elim ⇑s.inl ⇑s.inr, continuous …
    -/
    exacts [(ConcreteCategory.congr_hom h₁ x : _), (ConcreteCategory.congr_hom h₂ x : _)]
    /-
      🎉 no goals
    -/


theorem binaryCofan_isColimit_iff {X Y : TopCat} (c : BinaryCofan X Y) :
    Nonempty (IsColimit c) ↔
      IsOpenEmbedding c.inl ∧ IsOpenEmbedding c.inr ∧ IsCompl (range c.inl) (range c.inr) := by
  classical
    constructor
    · rintro ⟨h⟩
      rw [← show _ = c.inl from
          h.comp_coconePointUniqueUpToIso_inv (binaryCofanIsColimit X Y) ⟨WalkingPair.left⟩,
        ← show _ = c.inr from
          h.comp_coconePointUniqueUpToIso_inv (binaryCofanIsColimit X Y) ⟨WalkingPair.right⟩]
      dsimp
      refine ⟨(homeoOfIso <| h.coconePointUniqueUpToIso
        (binaryCofanIsColimit X Y)).symm.isOpenEmbedding.comp .inl,
          (homeoOfIso <| h.coconePointUniqueUpToIso
            (binaryCofanIsColimit X Y)).symm.isOpenEmbedding.comp .inr, ?_⟩
      erw [Set.range_comp, ← eq_compl_iff_isCompl, Set.range_comp _ Sum.inr,
        ← Set.image_compl_eq (homeoOfIso <| h.coconePointUniqueUpToIso
            (binaryCofanIsColimit X Y)).symm.bijective, Set.compl_range_inr, Set.image_comp]
    · rintro ⟨h₁, h₂, h₃⟩
      have : ∀ x, x ∈ Set.range c.inl ∨ x ∈ Set.range c.inr := by
        rw [eq_compl_iff_isCompl.mpr h₃.symm]
        exact fun _ => or_not
      refine ⟨BinaryCofan.IsColimit.mk _ ?_ ?_ ?_ ?_⟩
      · intro T f g
        refine ContinuousMap.mk ?_ ?_
        · exact fun x =>
            if h : x ∈ Set.range c.inl then f ((Equiv.ofInjective _ h₁.injective).symm ⟨x, h⟩)
            else g ((Equiv.ofInjective _ h₂.injective).symm ⟨x, (this x).resolve_left h⟩)
        rw [continuous_iff_continuousAt]
        intro x
        by_cases h : x ∈ Set.range c.inl
        · revert h x
          apply (IsOpen.continuousOn_iff _).mp
          · rw [continuousOn_iff_continuous_restrict]
            convert_to Continuous (f ∘ (Homeomorph.ofIsEmbedding _ h₁.isEmbedding).symm)
            · ext ⟨x, hx⟩
              exact dif_pos hx
            apply Continuous.comp
            · exact f.continuous_toFun
            · continuity
          · exact h₁.isOpen_range
        · revert h x
          apply (IsOpen.continuousOn_iff _).mp
          · rw [continuousOn_iff_continuous_restrict]
            have : ∀ a, a ∉ Set.range c.inl → a ∈ Set.range c.inr := by
              rintro a (h : a ∈ (Set.range c.inl)ᶜ)
              rwa [eq_compl_iff_isCompl.mpr h₃.symm]
            convert_to Continuous
                (g ∘ (Homeomorph.ofIsEmbedding _ h₂.isEmbedding).symm ∘ Subtype.map _ this)
            · ext ⟨x, hx⟩
              exact dif_neg hx
            apply Continuous.comp
            · exact g.continuous_toFun
            · apply Continuous.comp
              · continuity
              · rw [IsEmbedding.subtypeVal.isInducing.continuous_iff]
                exact continuous_subtype_val
          · change IsOpen (Set.range c.inl)ᶜ
            rw [← eq_compl_iff_isCompl.mpr h₃.symm]
            exact h₂.isOpen_range
      · intro T f g
        ext x
        refine (dif_pos ?_).trans ?_
        · exact ⟨x, rfl⟩
        · dsimp
          conv_lhs => rw [Equiv.ofInjective_symm_apply]
      · intro T f g
        ext x
        refine (dif_neg ?_).trans ?_
        · rintro ⟨y, e⟩
          have : c.inr x ∈ Set.range c.inl ⊓ Set.range c.inr := ⟨⟨_, e⟩, ⟨_, rfl⟩⟩
          rwa [disjoint_iff.mp h₃.1] at this
        · exact congr_arg g (Equiv.ofInjective_symm_apply _ _)
      · rintro T _ _ m rfl rfl
        ext x
        change m x = dite _ _ _
        split_ifs <;> exact congr_arg _ (Equiv.apply_ofInjective_symm _ ⟨_, _⟩).symm


