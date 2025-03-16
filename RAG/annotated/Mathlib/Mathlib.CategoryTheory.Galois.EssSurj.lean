private local instance fintypeQuotient (H : OpenSubgroup (G)) :
    Fintype (G ⧸ (H : Subgroup (G))) :=
  have : Finite (G ⧸ H.toSubgroup) := H.toSubgroup.quotient_finite_of_isOpen H.isOpen'
  Fintype.ofFinite _


private local instance fintypeQuotientStabilizer {X : Type*} [MulAction G X]
    [TopologicalSpace X] [ContinuousSMul G X] [DiscreteTopology X] (x : X) :
    Fintype (G ⧸ (MulAction.stabilizer (G) x)) :=
  fintypeQuotient ⟨MulAction.stabilizer (G) x, stabilizer_isOpen (G) x⟩


/-- If `X` is a finite discrete `G`-set, it can be written as the finite disjoint union
of quotients of the form `G ⧸ Uᵢ` for open subgroups `(Uᵢ)`. Note that this
is simply the decomposition into orbits. -/
lemma has_decomp_quotients (X : Action FintypeCat (MonCat.of G))
    [TopologicalSpace X.V] [DiscreteTopology X.V] [ContinuousSMul G X.V] :
    ∃ (ι : Type) (_ : Finite ι) (f : ι → OpenSubgroup (G)),
      Nonempty ((∐ fun i ↦ G ⧸ₐ (f i).toSubgroup) ≅ X) := by
  /-
    G : Type u_1
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalSpace G
    inst✝⁴ : TopologicalGroup G
    inst✝³ : CompactSpace G
    X : Action FintypeCat (MonCat.of G)
    inst✝² : TopologicalSpace ↑X.V
    inst✝¹ : DiscreteTopology ↑X.V
    inst✝ : ContinuousSMul G ↑X.V
    ⊢ Exists fun ι => Exists fun x => Exists fun f => Nonempty (CategoryTheory.Iso …
  -/
  obtain ⟨ι, hf, f, u, hc⟩ := has_decomp_connected_components' X
  /-
    case intro.intro.intro.intro
    G : Type u_1
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalSpace G
    inst✝⁴ : TopologicalGroup G
    inst✝³ : CompactSpace G
    X : Action FintypeCat (MonCat.of G)
    inst✝² : TopologicalSpace ↑X.V
    inst✝¹ : DiscreteTopology ↑X.V
    inst✝ : ContinuousSMul G ↑X.V
    ι : Type
    hf : Finite ι
    f : ι → Action FintypeCat (MonCat.of G)
    u : CategoryTheory.Iso (CategoryTheory.Limits.sigmaObj f) X
    hc : ∀ (i : ι), CategoryTheory.PreGaloisCategory.IsConnected (f i)
    ⊢ Exists fun ι => Exists fun x => Exists fun f => Nonempty (CategoryTheory.Iso …
  -/
  letI (i : ι) : TopologicalSpace (f i).V := ⊥
  /-
    case intro.intro.intro.intro
    G : Type u_1
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalSpace G
    inst✝⁴ : TopologicalGroup G
    inst✝³ : CompactSpace G
    X : Action FintypeCat (MonCat.of G)
    inst✝² : TopologicalSpace ↑X.V
    inst✝¹ : DiscreteTopology ↑X.V
    inst✝ : ContinuousSMul G ↑X.V
    ι : Type
    hf : Finite ι
    f : ι → Action FintypeCat (MonCat.of G)
    u : CategoryTheory.Iso (CategoryTheory.Limits.sigmaObj f) X
    hc : ∀ (i : ι), CategoryTheory.PreGaloisCategory.IsConnected (f i)
    this : (i : ι) → TopologicalSpace ↑(f i).V := fun i => Bot.bot
    ⊢ Exists fun ι => Exists fun x => Exists fun f => Nonempty (CategoryTheory.Iso …
  -/
  haveI (i : ι) : DiscreteTopology (f i).V := ⟨rfl⟩
  have (i : ι) : ContinuousSMul G (f i).V := ContinuousSMul.mk <| by
    let r : f i ⟶ X := Sigma.ι f i ≫ u.hom
    let r'' (p : G × (f i).V) : G × X.V := (p.1, r.hom p.2)
    let q (p : G × X.V) : X.V := X.ρ p.1 p.2
    let q' (p : G × (f i).V) : (f i).V := (f i).ρ p.1 p.2
    have heq : q ∘ r'' = r.hom ∘ q' := by
      ext (p : G × (f i).V)
      exact (congr_fun (r.comm p.1) p.2).symm
    have hrinj : Function.Injective r.hom :=
      (ConcreteCategory.mono_iff_injective_of_preservesPullback r).mp <| mono_comp _ _
    let t₁ : TopologicalSpace (G × (f i).V) := inferInstance
    show @Continuous _ _ _ ⊥ q'
    have : TopologicalSpace.induced r.hom inferInstance = ⊥ := by
      rw [← le_bot_iff]
      exact fun s _ ↦ ⟨r.hom '' s, ⟨isOpen_discrete (r.hom '' s), Set.preimage_image_eq s hrinj⟩⟩
    rw [← this, continuous_induced_rng, ← heq]
    exact Continuous.comp continuous_smul (by fun_prop)
  have (i : ι) : ∃ (U : OpenSubgroup (G)), (Nonempty ((f i) ≅ G ⧸ₐ U.toSubgroup)) := by
    obtain ⟨(x : (f i).V)⟩ := nonempty_fiber_of_isConnected (forget₂ _ _) (f i)
    let U : OpenSubgroup (G) := ⟨MulAction.stabilizer (G) x, stabilizer_isOpen (G) x⟩
    letI : Fintype (G ⧸ MulAction.stabilizer (G) x) := fintypeQuotient U
    exact ⟨U, ⟨FintypeCat.isoQuotientStabilizerOfIsConnected (f i) x⟩⟩
  /-
    case intro.intro.intro.intro
    G : Type u_1
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalSpace G
    inst✝⁴ : TopologicalGroup G
    inst✝³ : CompactSpace G
    X : Action FintypeCat (MonCat.of G)
    inst✝² : TopologicalSpace ↑X.V
    inst✝¹ : DiscreteTopology ↑X.V
    inst✝ : ContinuousSMul G ↑X.V
    ι : Type
    hf : Finite ι
    f : ι → Action FintypeCat (MonCat.of G)
    u : CategoryTheory.Iso (CategoryTheory.Limits.sigmaObj f) X
    hc : ∀ (i : ι), CategoryTheory.PreGaloisCategory.IsConnected (f i)
    this✝² : (i : ι) → TopologicalSpace ↑(f i).V := fun i => Bot.bot
    this✝¹ : ∀ (i : ι), DiscreteTopology ↑(f i).V
    this✝ : ∀ (i : ι), ContinuousSMul G ↑(f i).V
    this : ∀ (i : ι), Exists fun U => Nonempty (CategoryTheory.Iso (f i) (Action.F …
    ⊢ Exists fun ι => Exists fun x => Exists fun f => Nonempty (CategoryTheory.Iso …
  -/
  choose g ui using this
  /-
    case intro.intro.intro.intro
    G : Type u_1
    inst✝⁶ : Group G
    inst✝⁵ : TopologicalSpace G
    inst✝⁴ : TopologicalGroup G
    inst✝³ : CompactSpace G
    X : Action FintypeCat (MonCat.of G)
    inst✝² : TopologicalSpace ↑X.V
    inst✝¹ : DiscreteTopology ↑X.V
    inst✝ : ContinuousSMul G ↑X.V
    ι : Type
    hf : Finite ι
    f : ι → Action FintypeCat (MonCat.of G)
    u : CategoryTheory.Iso (CategoryTheory.Limits.sigmaObj f) X
    hc : ∀ (i : ι), CategoryTheory.PreGaloisCategory.IsConnected (f i)
    this✝¹ : (i : ι) → TopologicalSpace ↑(f i).V := fun i => Bot.bot
    this✝ : ∀ (i : ι), DiscreteTopology ↑(f i).V
    this : ∀ (i : ι), ContinuousSMul G ↑(f i).V
    g : ι → OpenSubgroup G
    ui : ∀ (i : ι), Nonempty (CategoryTheory.Iso (f i) (Action.FintypeCat.ofMulAct …
    ⊢ Exists fun ι => Exists fun x => Exists fun f => Nonempty (CategoryTheory.Iso …
  -/
  exact ⟨ι, hf, g, ⟨(Sigma.mapIso (fun i ↦ (ui i).some)).symm ≪≫ u⟩⟩
  /-
    🎉 no goals
  -/


/-- If `X` is connected and `x` is in the fiber of `X`, `F.obj X` is isomorphic
to the quotient of `Aut F` by the stabilizer of `x` as `Aut F`-sets. -/
def fiberIsoQuotientStabilizer (X : C) [IsConnected X] (x : F.obj X) :
    (functorToAction F).obj X ≅ Aut F ⧸ₐ MulAction.stabilizer (Aut F) x :=
  haveI : IsConnected ((functorToAction F).obj X) := PreservesIsConnected.preserves
  letI : Fintype (Aut F ⧸ MulAction.stabilizer (Aut F) x) := fintypeQuotientStabilizer x
  FintypeCat.isoQuotientStabilizerOfIsConnected ((functorToAction F).obj X) x


private def quotientToEndObjectHom :
    V.toSubgroup ⧸ Subgroup.subgroupOf U.toSubgroup V.toSubgroup →* End A :=
  let ff : (functorToAction F).FullyFaithful := FullyFaithful.ofFullyFaithful (functorToAction F)
  let e : End A ≃* End (Aut F ⧸ₐ U.toSubgroup) := (ff.mulEquivEnd A).trans (Iso.conj u)
  e.symm.toMonoidHom.comp (quotientToEndHom V.toSubgroup U.toSubgroup)


private lemma functorToAction_map_quotientToEndObjectHom
    (m : SingleObj.star (V ⧸ Subgroup.subgroupOf U.toSubgroup V.toSubgroup) ⟶
      SingleObj.star (V ⧸ Subgroup.subgroupOf U.toSubgroup V.toSubgroup)) :
    (functorToAction F).map (quotientToEndObjectHom V h u m) =
      u.hom ≫ quotientToEndHom V.toSubgroup U.toSubgroup m ≫ u.inv := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.GaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    V U : OpenSubgroup (CategoryTheory.Aut F)
    h : (↑U).Normal
    A : C
    u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
    m : Quiver.Hom (CategoryTheory.SingleObj.star (HasQuotient.Quotient (Subtype f …
    ⊢ Eq ((CategoryTheory.PreGaloisCategory.functorToAction F).map ((CategoryTheor …
  -/
  simp [← cancel_epi u.inv, ← cancel_mono u.hom, ← Iso.conj_apply, quotientToEndObjectHom]
  /-
    🎉 no goals
  -/


@[simps!]
private def quotientDiag : SingleObj (V.toSubgroup ⧸ Subgroup.subgroupOf U V) ⥤ C :=
  SingleObj.functor (quotientToEndObjectHom V h u)


@[simps]
private def coconeQuotientDiag :
    Cocone (quotientDiag V h u ⋙ functorToAction F) where
  pt := Aut F ⧸ₐ V.toSubgroup
  ι := SingleObj.natTrans (u.hom ≫ quotientToQuotientOfLE V.toSubgroup U.toSubgroup hUinV) <| by
    /-
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝⁵ : CategoryTheory.GaloisCategory C
      inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      G : Type u_1
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      V U : OpenSubgroup (CategoryTheory.Aut F)
      h : (↑U).Normal
      A : C
      u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      hUinV : LE.le U V
      ⊢ ∀ (a : HasQuotient.Quotient (Subtype fun x => Membership.mem (↑V) x) ((↑U).s …
    -/
    intro (m : V ⧸ Subgroup.subgroupOf U V)
    /-
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝⁵ : CategoryTheory.GaloisCategory C
      inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      G : Type u_1
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      V U : OpenSubgroup (CategoryTheory.Aut F)
      h : (↑U).Normal
      A : C
      u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      hUinV : LE.le U V
      m : HasQuotient.Quotient (Subtype fun x => Membership.mem V x) ((↑U).subgroupO …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.PreGaloisCategory.q …
    -/
    simp only [const_obj_obj, Functor.comp_map, const_obj_map, Category.comp_id]
    /-
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝⁵ : CategoryTheory.GaloisCategory C
      inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      G : Type u_1
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      V U : OpenSubgroup (CategoryTheory.Aut F)
      h : (↑U).Normal
      A : C
      u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      hUinV : LE.le U V
      m : HasQuotient.Quotient (Subtype fun x => Membership.mem V x) ((↑U).subgroupO …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.PreGaloisCategory.fu …
    -/
    rw [← cancel_epi (u.inv), Iso.inv_hom_id_assoc]
    /-
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝⁵ : CategoryTheory.GaloisCategory C
      inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      G : Type u_1
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      V U : OpenSubgroup (CategoryTheory.Aut F)
      h : (↑U).Normal
      A : C
      u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      hUinV : LE.le U V
      m : HasQuotient.Quotient (Subtype fun x => Membership.mem V x) ((↑U).subgroupO …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp u.inv (CategoryTheory.CategoryStruct. …
    -/
    apply Action.hom_ext
    /-
      case h
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝⁵ : CategoryTheory.GaloisCategory C
      inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      G : Type u_1
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      V U : OpenSubgroup (CategoryTheory.Aut F)
      h : (↑U).Normal
      A : C
      u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      hUinV : LE.le U V
      m : HasQuotient.Quotient (Subtype fun x => Membership.mem V x) ((↑U).subgroupO …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp u.inv (CategoryTheory.CategoryStruct. …
    -/
    ext (x : Aut F ⧸ U.toSubgroup)
    /-
      case h.h
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝⁵ : CategoryTheory.GaloisCategory C
      inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      G : Type u_1
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      V U : OpenSubgroup (CategoryTheory.Aut F)
      h : (↑U).Normal
      A : C
      u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      hUinV : LE.le U V
      m : HasQuotient.Quotient (Subtype fun x => Membership.mem V x) ((↑U).subgroupO …
      x : HasQuotient.Quotient (CategoryTheory.Aut F) ↑U
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp u.inv (CategoryTheory.CategoryStruct …
    -/
    induction' m, x using Quotient.inductionOn₂ with σ μ
    suffices h : ⟦μ * σ⁻¹⟧ = ⟦μ⟧ by
      simp only [quotientToQuotientOfLE_hom_mk, quotientDiag_map,
        functorToAction_map_quotientToEndObjectHom V _ u]
      simpa
    /-
      case h.h.h
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝⁵ : CategoryTheory.GaloisCategory C
      inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      G : Type u_1
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      V U : OpenSubgroup (CategoryTheory.Aut F)
      h : (↑U).Normal
      A : C
      u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      hUinV : LE.le U V
      σ : Subtype fun x => Membership.mem (↑V) x
      μ : CategoryTheory.Aut F
      ⊢ Eq (Quotient.mk (QuotientGroup.leftRel ↑V) (HMul.hMul μ ↑(Inv.inv σ))) (Quot …
    -/
    apply Quotient.sound
    /-
      case h.h.h.a
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝⁵ : CategoryTheory.GaloisCategory C
      inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      G : Type u_1
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      V U : OpenSubgroup (CategoryTheory.Aut F)
      h : (↑U).Normal
      A : C
      u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      hUinV : LE.le U V
      σ : Subtype fun x => Membership.mem (↑V) x
      μ : CategoryTheory.Aut F
      ⊢ HasEquiv.Equiv (HMul.hMul μ ↑(Inv.inv σ)) μ
    -/
    apply (QuotientGroup.leftRel_apply).mpr
    /-
      case h.h.h.a
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝⁵ : CategoryTheory.GaloisCategory C
      inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      G : Type u_1
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      V U : OpenSubgroup (CategoryTheory.Aut F)
      h : (↑U).Normal
      A : C
      u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      hUinV : LE.le U V
      σ : Subtype fun x => Membership.mem (↑V) x
      μ : CategoryTheory.Aut F
      ⊢ Membership.mem (↑V) (HMul.hMul (Inv.inv (HMul.hMul μ ↑(Inv.inv σ))) μ)
    -/
    simp
    /-
      🎉 no goals
    -/


@[simps]
private def coconeQuotientDiagDesc
    (s : Cocone (quotientDiag V h u ⋙ functorToAction F)) :
      (coconeQuotientDiag h u hUinV).pt ⟶ s.pt where
  hom := Quotient.lift (fun σ ↦ (u.inv ≫ s.ι.app (SingleObj.star _)).hom ⟦σ⟧) <| fun σ τ hst ↦ by
    /-
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝⁵ : CategoryTheory.GaloisCategory C
      inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      G : Type u_1
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      V U : OpenSubgroup (CategoryTheory.Aut F)
      h : (↑U).Normal
      A : C
      u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      hUinV : LE.le U V
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.PreGaloisCategory.quotientDi …
      σ τ : CategoryTheory.Aut F
      hst : HasEquiv.Equiv σ τ
      ⊢ Eq ((fun σ => (CategoryTheory.CategoryStruct.comp u.inv (s.ι.app (CategoryTh …
    -/
    let J' := quotientDiag V h u ⋙ functorToAction F
    let m : End (SingleObj.star (V.toSubgroup ⧸ Subgroup.subgroupOf U V)) :=
      ⟦⟨σ⁻¹ * τ, (QuotientGroup.leftRel_apply).mp hst⟩⟧
    /-
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝⁵ : CategoryTheory.GaloisCategory C
      inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      G : Type u_1
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      V U : OpenSubgroup (CategoryTheory.Aut F)
      h : (↑U).Normal
      A : C
      u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      hUinV : LE.le U V
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.PreGaloisCategory.quotientDi …
      σ τ : CategoryTheory.Aut F
      hst : HasEquiv.Equiv σ τ
      J' : CategoryTheory.Functor (CategoryTheory.SingleObj (HasQuotient.Quotient (S …
      m : CategoryTheory.End (CategoryTheory.SingleObj.star (HasQuotient.Quotient (S …
      ⊢ Eq ((fun σ => (CategoryTheory.CategoryStruct.comp u.inv (s.ι.app (CategoryTh …
    -/
    have h1 : J'.map m ≫ s.ι.app (SingleObj.star _) = s.ι.app (SingleObj.star _) := s.ι.naturality m
    /-
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝⁵ : CategoryTheory.GaloisCategory C
      inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      G : Type u_1
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      V U : OpenSubgroup (CategoryTheory.Aut F)
      h : (↑U).Normal
      A : C
      u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      hUinV : LE.le U V
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.PreGaloisCategory.quotientDi …
      σ τ : CategoryTheory.Aut F
      hst : HasEquiv.Equiv σ τ
      J' : CategoryTheory.Functor (CategoryTheory.SingleObj (HasQuotient.Quotient (S …
      m : CategoryTheory.End (CategoryTheory.SingleObj.star (HasQuotient.Quotient (S …
      h1 : Eq (CategoryTheory.CategoryStruct.comp (J'.map m) (s.ι.app (CategoryTheor …
      ⊢ Eq ((fun σ => (CategoryTheory.CategoryStruct.comp u.inv (s.ι.app (CategoryTh …
    -/
    conv_rhs => rw [← h1]
    have h2 : (J'.map m).hom (u.inv.hom ⟦τ⟧) = u.inv.hom ⟦σ⟧ := by
      simp only [comp_obj, quotientDiag_obj, Functor.comp_map, quotientDiag_map, J',
        functorToAction_map_quotientToEndObjectHom V h u m]
      show (u.inv ≫ u.hom ≫ _ ≫ u.inv).hom ⟦τ⟧ = u.inv.hom ⟦σ⟧
      simp [m]
    /-
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝⁵ : CategoryTheory.GaloisCategory C
      inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      G : Type u_1
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      V U : OpenSubgroup (CategoryTheory.Aut F)
      h : (↑U).Normal
      A : C
      u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      hUinV : LE.le U V
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.PreGaloisCategory.quotientDi …
      σ τ : CategoryTheory.Aut F
      hst : HasEquiv.Equiv σ τ
      J' : CategoryTheory.Functor (CategoryTheory.SingleObj (HasQuotient.Quotient (S …
      m : CategoryTheory.End (CategoryTheory.SingleObj.star (HasQuotient.Quotient (S …
      h1 : Eq (CategoryTheory.CategoryStruct.comp (J'.map m) (s.ι.app (CategoryTheor …
      h2 : Eq ((J'.map m).hom (u.inv.hom (Quotient.mk (QuotientGroup.leftRel ↑U) τ)) …
      ⊢ Eq ((fun σ => (CategoryTheory.CategoryStruct.comp u.inv (s.ι.app (CategoryTh …
    -/
    simp only [← h2, const_obj_obj, Action.comp_hom, FintypeCat.comp_apply]
    /-
      🎉 no goals
    -/
  comm g := by
    /-
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝⁵ : CategoryTheory.GaloisCategory C
      inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      G : Type u_1
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      V U : OpenSubgroup (CategoryTheory.Aut F)
      h : (↑U).Normal
      A : C
      u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      hUinV : LE.le U V
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.PreGaloisCategory.quotientDi …
      g : ↑(MonCat.of (CategoryTheory.Aut F))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.PreGaloisCategory.co …
    -/
    ext (x : Aut F ⧸ V.toSubgroup)
    /-
      case h
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝⁵ : CategoryTheory.GaloisCategory C
      inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      G : Type u_1
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      V U : OpenSubgroup (CategoryTheory.Aut F)
      h : (↑U).Normal
      A : C
      u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      hUinV : LE.le U V
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.PreGaloisCategory.quotientDi …
      g : ↑(MonCat.of (CategoryTheory.Aut F))
      x : HasQuotient.Quotient (CategoryTheory.Aut F) ↑V
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.PreGaloisCategory.co …
    -/
    induction' x using Quotient.inductionOn with σ
    /-
      case h.h
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝⁵ : CategoryTheory.GaloisCategory C
      inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      G : Type u_1
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      V U : OpenSubgroup (CategoryTheory.Aut F)
      h : (↑U).Normal
      A : C
      u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      hUinV : LE.le U V
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.PreGaloisCategory.quotientDi …
      g : ↑(MonCat.of (CategoryTheory.Aut F))
      σ : CategoryTheory.Aut F
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.PreGaloisCategory.co …
    -/
    simp only [const_obj_obj]
    show (((Aut F ⧸ₐ U.toSubgroup).ρ g ≫ u.inv.hom) ≫ (s.ι.app (SingleObj.star _)).hom) ⟦σ⟧ =
      ((s.ι.app (SingleObj.star _)).hom ≫ s.pt.ρ g) (u.inv.hom ⟦σ⟧)
    have : ((functorToAction F).obj A).ρ g ≫ (s.ι.app (SingleObj.star _)).hom =
        (s.ι.app (SingleObj.star _)).hom ≫ s.pt.ρ g :=
      (s.ι.app (SingleObj.star _)).comm g
    /-
      case h.h
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝⁵ : CategoryTheory.GaloisCategory C
      inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      G : Type u_1
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      V U : OpenSubgroup (CategoryTheory.Aut F)
      h : (↑U).Normal
      A : C
      u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      hUinV : LE.le U V
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.PreGaloisCategory.quotientDi …
      g : ↑(MonCat.of (CategoryTheory.Aut F))
      σ : CategoryTheory.Aut F
      this : Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.PreGaloisCateg …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [← this, u.inv.comm g]
    /-
      case h.h
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝⁵ : CategoryTheory.GaloisCategory C
      inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      G : Type u_1
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      V U : OpenSubgroup (CategoryTheory.Aut F)
      h : (↑U).Normal
      A : C
      u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      hUinV : LE.le U V
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.PreGaloisCategory.quotientDi …
      g : ↑(MonCat.of (CategoryTheory.Aut F))
      σ : CategoryTheory.Aut F
      this : Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.PreGaloisCateg …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp u …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- The constructed cocone `coconeQuotientDiag` on the diagram `quotientDiag` is colimiting. -/
private def coconeQuotientDiagIsColimit :
    IsColimit (coconeQuotientDiag h u hUinV) where
  desc := coconeQuotientDiagDesc h u hUinV
  fac s j := by
    /-
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝⁵ : CategoryTheory.GaloisCategory C
      inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      G : Type u_1
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      V U : OpenSubgroup (CategoryTheory.Aut F)
      h : (↑U).Normal
      A : C
      u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      hUinV : LE.le U V
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.PreGaloisCategory.quotientDi …
      j : CategoryTheory.SingleObj (HasQuotient.Quotient (Subtype fun x => Membershi …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.PreGaloisCategory.co …
    -/
    apply (cancel_epi u.inv).mp
    /-
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝⁵ : CategoryTheory.GaloisCategory C
      inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      G : Type u_1
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      V U : OpenSubgroup (CategoryTheory.Aut F)
      h : (↑U).Normal
      A : C
      u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      hUinV : LE.le U V
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.PreGaloisCategory.quotientDi …
      j : CategoryTheory.SingleObj (HasQuotient.Quotient (Subtype fun x => Membershi …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp u.inv (CategoryTheory.CategoryStruct. …
    -/
    apply Action.hom_ext
    /-
      case h
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝⁵ : CategoryTheory.GaloisCategory C
      inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      G : Type u_1
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      V U : OpenSubgroup (CategoryTheory.Aut F)
      h : (↑U).Normal
      A : C
      u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      hUinV : LE.le U V
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.PreGaloisCategory.quotientDi …
      j : CategoryTheory.SingleObj (HasQuotient.Quotient (Subtype fun x => Membershi …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp u.inv (CategoryTheory.CategoryStruct. …
    -/
    ext (x : Aut F ⧸ U.toSubgroup)
    /-
      case h.h
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝⁵ : CategoryTheory.GaloisCategory C
      inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      G : Type u_1
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      V U : OpenSubgroup (CategoryTheory.Aut F)
      h : (↑U).Normal
      A : C
      u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      hUinV : LE.le U V
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.PreGaloisCategory.quotientDi …
      j : CategoryTheory.SingleObj (HasQuotient.Quotient (Subtype fun x => Membershi …
      x : HasQuotient.Quotient (CategoryTheory.Aut F) ↑U
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp u.inv (CategoryTheory.CategoryStruct …
    -/
    induction' x using Quotient.inductionOn with σ
    /-
      case h.h.h
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝⁵ : CategoryTheory.GaloisCategory C
      inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      G : Type u_1
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      V U : OpenSubgroup (CategoryTheory.Aut F)
      h : (↑U).Normal
      A : C
      u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      hUinV : LE.le U V
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.PreGaloisCategory.quotientDi …
      j : CategoryTheory.SingleObj (HasQuotient.Quotient (Subtype fun x => Membershi …
      σ : CategoryTheory.Aut F
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp u.inv (CategoryTheory.CategoryStruct …
    -/
    simp
    /-
      case h.h.h
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝⁵ : CategoryTheory.GaloisCategory C
      inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      G : Type u_1
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      V U : OpenSubgroup (CategoryTheory.Aut F)
      h : (↑U).Normal
      A : C
      u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      hUinV : LE.le U V
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.PreGaloisCategory.quotientDi …
      j : CategoryTheory.SingleObj (HasQuotient.Quotient (Subtype fun x => Membershi …
      σ : CategoryTheory.Aut F
      ⊢ Eq ((s.ι.app (CategoryTheory.SingleObj.star (HasQuotient.Quotient (Subtype f …
    -/
    rfl
    /-
      🎉 no goals
    -/
  uniq s f hf := by
    /-
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝⁵ : CategoryTheory.GaloisCategory C
      inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      G : Type u_1
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      V U : OpenSubgroup (CategoryTheory.Aut F)
      h : (↑U).Normal
      A : C
      u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      hUinV : LE.le U V
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.PreGaloisCategory.quotientDi …
      f : Quiver.Hom (CategoryTheory.PreGaloisCategory.coconeQuotientDiag h u hUinV) …
      hf : ∀ (j : CategoryTheory.SingleObj (HasQuotient.Quotient (Subtype fun x => M …
      ⊢ Eq f (CategoryTheory.PreGaloisCategory.coconeQuotientDiagDesc h u hUinV s)
    -/
    apply Action.hom_ext
    /-
      case h
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝⁵ : CategoryTheory.GaloisCategory C
      inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      G : Type u_1
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      V U : OpenSubgroup (CategoryTheory.Aut F)
      h : (↑U).Normal
      A : C
      u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      hUinV : LE.le U V
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.PreGaloisCategory.quotientDi …
      f : Quiver.Hom (CategoryTheory.PreGaloisCategory.coconeQuotientDiag h u hUinV) …
      hf : ∀ (j : CategoryTheory.SingleObj (HasQuotient.Quotient (Subtype fun x => M …
      ⊢ Eq f.hom (CategoryTheory.PreGaloisCategory.coconeQuotientDiagDesc h u hUinV  …
    -/
    ext (x : Aut F ⧸ V.toSubgroup)
    /-
      case h.h
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝⁵ : CategoryTheory.GaloisCategory C
      inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      G : Type u_1
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      V U : OpenSubgroup (CategoryTheory.Aut F)
      h : (↑U).Normal
      A : C
      u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      hUinV : LE.le U V
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.PreGaloisCategory.quotientDi …
      f : Quiver.Hom (CategoryTheory.PreGaloisCategory.coconeQuotientDiag h u hUinV) …
      hf : ∀ (j : CategoryTheory.SingleObj (HasQuotient.Quotient (Subtype fun x => M …
      x : HasQuotient.Quotient (CategoryTheory.Aut F) ↑V
      ⊢ Eq (f.hom x) ((CategoryTheory.PreGaloisCategory.coconeQuotientDiagDesc h u h …
    -/
    induction' x using Quotient.inductionOn with σ
    /-
      case h.h.h
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{u₂, u₁} C
      F : CategoryTheory.Functor C FintypeCat
      inst✝⁵ : CategoryTheory.GaloisCategory C
      inst✝⁴ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      G : Type u_1
      inst✝³ : Group G
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalGroup G
      inst✝ : CompactSpace G
      V U : OpenSubgroup (CategoryTheory.Aut F)
      h : (↑U).Normal
      A : C
      u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
      hUinV : LE.le U V
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.PreGaloisCategory.quotientDi …
      f : Quiver.Hom (CategoryTheory.PreGaloisCategory.coconeQuotientDiag h u hUinV) …
      hf : ∀ (j : CategoryTheory.SingleObj (HasQuotient.Quotient (Subtype fun x => M …
      σ : CategoryTheory.Aut F
      ⊢ Eq (f.hom (Quotient.mk (QuotientGroup.leftRel ↑V) σ)) ((CategoryTheory.PreGa …
    -/
    simp [← hf (SingleObj.star _)]
    /-
      🎉 no goals
    -/


/-- For every open subgroup `V` of `Aut F`, there exists an `X : C` such that
`F.obj X ≅ Aut F ⧸ V` as `Aut F`-sets. -/
lemma exists_lift_of_quotient_openSubgroup (V : OpenSubgroup (Aut F)) :
    ∃ (X : C), Nonempty ((functorToAction F).obj X ≅ Aut F ⧸ₐ V.toSubgroup) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.GaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    V : OpenSubgroup (CategoryTheory.Aut F)
    ⊢ Exists fun X => Nonempty (CategoryTheory.Iso ((CategoryTheory.PreGaloisCateg …
  -/
  obtain ⟨I, hf, hc, hi⟩ := exists_set_ker_evaluation_subset_of_isOpen F (one_mem V) V.isOpen'
  /-
    case intro.intro.intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.GaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    V : OpenSubgroup (CategoryTheory.Aut F)
    I : Set C
    hf : Fintype ↑I
    hc : ∀ (X : C), Membership.mem I X → CategoryTheory.PreGaloisCategory.IsConnec …
    hi : ∀ (σ : CategoryTheory.Aut F), (∀ (X : ↑I), Eq (σ.hom.app ↑X) (CategoryThe …
    ⊢ Exists fun X => Nonempty (CategoryTheory.Iso ((CategoryTheory.PreGaloisCateg …
  -/
  haveI (X : I) : IsConnected X.val := hc X X.property
  /-
    case intro.intro.intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.GaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    V : OpenSubgroup (CategoryTheory.Aut F)
    I : Set C
    hf : Fintype ↑I
    hc : ∀ (X : C), Membership.mem I X → CategoryTheory.PreGaloisCategory.IsConnec …
    hi : ∀ (σ : CategoryTheory.Aut F), (∀ (X : ↑I), Eq (σ.hom.app ↑X) (CategoryThe …
    this : ∀ (X : ↑I), CategoryTheory.PreGaloisCategory.IsConnected ↑X
    ⊢ Exists fun X => Nonempty (CategoryTheory.Iso ((CategoryTheory.PreGaloisCateg …
  -/
  haveI (X : I) : Nonempty (F.obj X.val) := nonempty_fiber_of_isConnected F X
  /-
    case intro.intro.intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.GaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    V : OpenSubgroup (CategoryTheory.Aut F)
    I : Set C
    hf : Fintype ↑I
    hc : ∀ (X : C), Membership.mem I X → CategoryTheory.PreGaloisCategory.IsConnec …
    hi : ∀ (σ : CategoryTheory.Aut F), (∀ (X : ↑I), Eq (σ.hom.app ↑X) (CategoryThe …
    this✝ : ∀ (X : ↑I), CategoryTheory.PreGaloisCategory.IsConnected ↑X
    this : ∀ (X : ↑I), Nonempty ↑(F.obj ↑X)
    ⊢ Exists fun X => Nonempty (CategoryTheory.Iso ((CategoryTheory.PreGaloisCateg …
  -/
  have hn : Nonempty (F.obj <| (∏ᶜ fun X : I => X)) := nonempty_fiber_pi_of_nonempty_of_finite F _
  /-
    case intro.intro.intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.GaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    V : OpenSubgroup (CategoryTheory.Aut F)
    I : Set C
    hf : Fintype ↑I
    hc : ∀ (X : C), Membership.mem I X → CategoryTheory.PreGaloisCategory.IsConnec …
    hi : ∀ (σ : CategoryTheory.Aut F), (∀ (X : ↑I), Eq (σ.hom.app ↑X) (CategoryThe …
    this✝ : ∀ (X : ↑I), CategoryTheory.PreGaloisCategory.IsConnected ↑X
    this : ∀ (X : ↑I), Nonempty ↑(F.obj ↑X)
    hn : Nonempty ↑(F.obj (CategoryTheory.Limits.piObj fun X => ↑X))
    ⊢ Exists fun X => Nonempty (CategoryTheory.Iso ((CategoryTheory.PreGaloisCateg …
  -/
  obtain ⟨A, f, hgal⟩ := exists_hom_from_galois_of_fiber_nonempty F (∏ᶜ fun X : I => X) hn
  /-
    case intro.intro.intro.intro.intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.GaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    V : OpenSubgroup (CategoryTheory.Aut F)
    I : Set C
    hf : Fintype ↑I
    hc : ∀ (X : C), Membership.mem I X → CategoryTheory.PreGaloisCategory.IsConnec …
    hi : ∀ (σ : CategoryTheory.Aut F), (∀ (X : ↑I), Eq (σ.hom.app ↑X) (CategoryThe …
    this✝ : ∀ (X : ↑I), CategoryTheory.PreGaloisCategory.IsConnected ↑X
    this : ∀ (X : ↑I), Nonempty ↑(F.obj ↑X)
    hn : Nonempty ↑(F.obj (CategoryTheory.Limits.piObj fun X => ↑X))
    A : C
    f : Quiver.Hom A (CategoryTheory.Limits.piObj fun X => ↑X)
    hgal : CategoryTheory.PreGaloisCategory.IsGalois A
    ⊢ Exists fun X => Nonempty (CategoryTheory.Iso ((CategoryTheory.PreGaloisCateg …
  -/
  obtain ⟨a⟩ := nonempty_fiber_of_isConnected F A
  /-
    case intro.intro.intro.intro.intro.intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.GaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    V : OpenSubgroup (CategoryTheory.Aut F)
    I : Set C
    hf : Fintype ↑I
    hc : ∀ (X : C), Membership.mem I X → CategoryTheory.PreGaloisCategory.IsConnec …
    hi : ∀ (σ : CategoryTheory.Aut F), (∀ (X : ↑I), Eq (σ.hom.app ↑X) (CategoryThe …
    this✝ : ∀ (X : ↑I), CategoryTheory.PreGaloisCategory.IsConnected ↑X
    this : ∀ (X : ↑I), Nonempty ↑(F.obj ↑X)
    hn : Nonempty ↑(F.obj (CategoryTheory.Limits.piObj fun X => ↑X))
    A : C
    f : Quiver.Hom A (CategoryTheory.Limits.piObj fun X => ↑X)
    hgal : CategoryTheory.PreGaloisCategory.IsGalois A
    a : ↑(F.obj A)
    ⊢ Exists fun X => Nonempty (CategoryTheory.Iso ((CategoryTheory.PreGaloisCateg …
  -/
  let U : OpenSubgroup (Aut F) := ⟨MulAction.stabilizer (Aut F) a, stabilizer_isOpen (Aut F) a⟩
  /-
    case intro.intro.intro.intro.intro.intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.GaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    V : OpenSubgroup (CategoryTheory.Aut F)
    I : Set C
    hf : Fintype ↑I
    hc : ∀ (X : C), Membership.mem I X → CategoryTheory.PreGaloisCategory.IsConnec …
    hi : ∀ (σ : CategoryTheory.Aut F), (∀ (X : ↑I), Eq (σ.hom.app ↑X) (CategoryThe …
    this✝ : ∀ (X : ↑I), CategoryTheory.PreGaloisCategory.IsConnected ↑X
    this : ∀ (X : ↑I), Nonempty ↑(F.obj ↑X)
    hn : Nonempty ↑(F.obj (CategoryTheory.Limits.piObj fun X => ↑X))
    A : C
    f : Quiver.Hom A (CategoryTheory.Limits.piObj fun X => ↑X)
    hgal : CategoryTheory.PreGaloisCategory.IsGalois A
    a : ↑(F.obj A)
    U : OpenSubgroup (CategoryTheory.Aut F) := { toSubgroup := MulAction.stabilize …
    ⊢ Exists fun X => Nonempty (CategoryTheory.Iso ((CategoryTheory.PreGaloisCateg …
  -/
  let u := fiberIsoQuotientStabilizer A a
  /-
    case intro.intro.intro.intro.intro.intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.GaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    V : OpenSubgroup (CategoryTheory.Aut F)
    I : Set C
    hf : Fintype ↑I
    hc : ∀ (X : C), Membership.mem I X → CategoryTheory.PreGaloisCategory.IsConnec …
    hi : ∀ (σ : CategoryTheory.Aut F), (∀ (X : ↑I), Eq (σ.hom.app ↑X) (CategoryThe …
    this✝ : ∀ (X : ↑I), CategoryTheory.PreGaloisCategory.IsConnected ↑X
    this : ∀ (X : ↑I), Nonempty ↑(F.obj ↑X)
    hn : Nonempty ↑(F.obj (CategoryTheory.Limits.piObj fun X => ↑X))
    A : C
    f : Quiver.Hom A (CategoryTheory.Limits.piObj fun X => ↑X)
    hgal : CategoryTheory.PreGaloisCategory.IsGalois A
    a : ↑(F.obj A)
    U : OpenSubgroup (CategoryTheory.Aut F) := { toSubgroup := MulAction.stabilize …
    u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
    ⊢ Exists fun X => Nonempty (CategoryTheory.Iso ((CategoryTheory.PreGaloisCateg …
  -/
  have hUnormal : U.toSubgroup.Normal := stabilizer_normal_of_isGalois F A a
  have h1 (σ : Aut F) (σinU : σ ∈ U) : σ.hom.app A = 𝟙 (F.obj A) := by
    have hi : (Aut F ⧸ₐ MulAction.stabilizer (Aut F) a).ρ σ = 𝟙 _ := by
      refine FintypeCat.hom_ext _ _ (fun x ↦ ?_)
      induction' x using Quotient.inductionOn with τ
      show ⟦σ * τ⟧ = ⟦τ⟧
      apply Quotient.sound
      apply (QuotientGroup.leftRel_apply).mpr
      simp only [mul_inv_rev]
      exact Subgroup.Normal.conj_mem hUnormal _ (Subgroup.inv_mem U.toSubgroup σinU) _
    simp [← cancel_mono u.hom.hom, show σ.hom.app A ≫ u.hom.hom = _ from u.hom.comm σ, hi]
  have h2 (σ : Aut F) (σinU : σ ∈ U) : ∀ X : I, σ.hom.app X = 𝟙 (F.obj X) := by
    intro ⟨X, hX⟩
    ext (x : F.obj X)
    let p : A ⟶ X := f ≫ Pi.π (fun Z : I => (Z : C)) ⟨X, hX⟩
    have : IsConnected X := hc X hX
    obtain ⟨a, rfl⟩ := surjective_of_nonempty_fiber_of_isConnected F p x
    simp only [FintypeCat.id_apply, FunctorToFintypeCat.naturality, h1 σ σinU]
  /-
    case intro.intro.intro.intro.intro.intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.GaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    V : OpenSubgroup (CategoryTheory.Aut F)
    I : Set C
    hf : Fintype ↑I
    hc : ∀ (X : C), Membership.mem I X → CategoryTheory.PreGaloisCategory.IsConnec …
    hi : ∀ (σ : CategoryTheory.Aut F), (∀ (X : ↑I), Eq (σ.hom.app ↑X) (CategoryThe …
    this✝ : ∀ (X : ↑I), CategoryTheory.PreGaloisCategory.IsConnected ↑X
    this : ∀ (X : ↑I), Nonempty ↑(F.obj ↑X)
    hn : Nonempty ↑(F.obj (CategoryTheory.Limits.piObj fun X => ↑X))
    A : C
    f : Quiver.Hom A (CategoryTheory.Limits.piObj fun X => ↑X)
    hgal : CategoryTheory.PreGaloisCategory.IsGalois A
    a : ↑(F.obj A)
    U : OpenSubgroup (CategoryTheory.Aut F) := { toSubgroup := MulAction.stabilize …
    u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
    hUnormal : (↑U).Normal
    h1 : ∀ (σ : CategoryTheory.Aut F), Membership.mem U σ → Eq (σ.hom.app A) (Cate …
    h2 : ∀ (σ : CategoryTheory.Aut F), Membership.mem U σ → ∀ (X : ↑I), Eq (σ.hom. …
    ⊢ Exists fun X => Nonempty (CategoryTheory.Iso ((CategoryTheory.PreGaloisCateg …
  -/
  have hUinV : (U : Set (Aut F)) ≤ V := fun u uinU ↦ hi u (h2 u uinU)
  /-
    case intro.intro.intro.intro.intro.intro
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.GaloisCategory C
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    V : OpenSubgroup (CategoryTheory.Aut F)
    I : Set C
    hf : Fintype ↑I
    hc : ∀ (X : C), Membership.mem I X → CategoryTheory.PreGaloisCategory.IsConnec …
    hi : ∀ (σ : CategoryTheory.Aut F), (∀ (X : ↑I), Eq (σ.hom.app ↑X) (CategoryThe …
    this✝ : ∀ (X : ↑I), CategoryTheory.PreGaloisCategory.IsConnected ↑X
    this : ∀ (X : ↑I), Nonempty ↑(F.obj ↑X)
    hn : Nonempty ↑(F.obj (CategoryTheory.Limits.piObj fun X => ↑X))
    A : C
    f : Quiver.Hom A (CategoryTheory.Limits.piObj fun X => ↑X)
    hgal : CategoryTheory.PreGaloisCategory.IsGalois A
    a : ↑(F.obj A)
    U : OpenSubgroup (CategoryTheory.Aut F) := { toSubgroup := MulAction.stabilize …
    u : CategoryTheory.Iso ((CategoryTheory.PreGaloisCategory.functorToAction F).o …
    hUnormal : (↑U).Normal
    h1 : ∀ (σ : CategoryTheory.Aut F), Membership.mem U σ → Eq (σ.hom.app A) (Cate …
    h2 : ∀ (σ : CategoryTheory.Aut F), Membership.mem U σ → ∀ (X : ↑I), Eq (σ.hom. …
    hUinV : LE.le ↑U ↑V
    ⊢ Exists fun X => Nonempty (CategoryTheory.Iso ((CategoryTheory.PreGaloisCateg …
  -/
  have := V.quotient_finite_of_isOpen' (U.subgroupOf V) V.isOpen (V.subgroupOf_isOpen U U.isOpen)
  exact ⟨colimit (quotientDiag V hUnormal u),
    ⟨preservesColimitIso (functorToAction F) (quotientDiag V hUnormal u) ≪≫
    colimit.isoColimitCocone ⟨coconeQuotientDiag hUnormal u hUinV,
    coconeQuotientDiagIsColimit hUnormal u hUinV⟩⟩⟩


/--
If `X` is a finite, discrete `Aut F`-set with continuous `Aut F`-action, then
there exists `A : C` such that `F.obj A ≅ X` as `Aut F`-sets.
-/
@[stacks 0BN4 "Essential surjectivity part"]
theorem exists_lift_of_continuous (X : Action FintypeCat (MonCat.of (Aut F)))
    [TopologicalSpace X.V] [DiscreteTopology X.V] [ContinuousSMul (Aut F) X.V] :
    ∃ A, Nonempty ((functorToAction F).obj A ≅ X) := by
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝⁴ : CategoryTheory.GaloisCategory C
    inst✝³ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : Action FintypeCat (MonCat.of (CategoryTheory.Aut F))
    inst✝² : TopologicalSpace ↑X.V
    inst✝¹ : DiscreteTopology ↑X.V
    inst✝ : ContinuousSMul (CategoryTheory.Aut F) ↑X.V
    ⊢ Exists fun A => Nonempty (CategoryTheory.Iso ((CategoryTheory.PreGaloisCateg …
  -/
  obtain ⟨ι, hfin, f, ⟨u⟩⟩ := has_decomp_quotients X
  /-
    case intro.intro.intro.intro
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{u₂, u₁} C
    F : CategoryTheory.Functor C FintypeCat
    inst✝⁴ : CategoryTheory.GaloisCategory C
    inst✝³ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : Action FintypeCat (MonCat.of (CategoryTheory.Aut F))
    inst✝² : TopologicalSpace ↑X.V
    inst✝¹ : DiscreteTopology ↑X.V
    inst✝ : ContinuousSMul (CategoryTheory.Aut F) ↑X.V
    ι : Type
    hfin : Finite ι
    f : ι → OpenSubgroup (CategoryTheory.Aut F)
    u : CategoryTheory.Iso (CategoryTheory.Limits.sigmaObj fun i => Action.Fintype …
    ⊢ Exists fun A => Nonempty (CategoryTheory.Iso ((CategoryTheory.PreGaloisCateg …
  -/
  choose g gu using (fun i ↦ exists_lift_of_quotient_openSubgroup (f i))
  exact ⟨∐ g, ⟨PreservesCoproduct.iso (functorToAction F) g ≪≫
    Sigma.mapIso (fun i ↦ (gu i).some) ≪≫ u⟩⟩


