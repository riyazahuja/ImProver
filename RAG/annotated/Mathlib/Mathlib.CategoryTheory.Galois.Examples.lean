/-- Complement of the image of a morphism `f : X ⟶ Y` in `FintypeCat`. -/
noncomputable def imageComplement {X Y : FintypeCat.{u}} (f : X ⟶ Y) :
    FintypeCat.{u} := by
  /-
    X Y : FintypeCat
    f : Quiver.Hom X Y
    ⊢ FintypeCat
  -/
  haveI : Fintype (↑(Set.range f)ᶜ) := Fintype.ofFinite _
  /-
    X Y : FintypeCat
    f : Quiver.Hom X Y
    this : Fintype ↑(HasCompl.compl (Set.range f))
    ⊢ FintypeCat
  -/
  exact FintypeCat.of (↑(Set.range f)ᶜ)
  /-
    🎉 no goals
  -/


/-- The inclusion from the complement of the image of `f : X ⟶ Y` into `Y`. -/
def imageComplementIncl {X Y : FintypeCat.{u}}
    (f : X ⟶ Y) : imageComplement f ⟶ Y :=
  Subtype.val


/-- Given `f : X ⟶ Y` for `X Y : Action FintypeCat (MonCat.of G)`, the complement of the image
of `f` has a natural `G`-action. -/
noncomputable def Action.imageComplement {X Y : Action FintypeCat (MonCat.of G)}
    (f : X ⟶ Y) : Action FintypeCat (MonCat.of G) where
  V := FintypeCat.imageComplement f.hom
  ρ := MonCat.ofHom <| {
    toFun := fun g y ↦ Subtype.mk (Y.ρ g y.val) <| by
      /-
        G : Type u
        inst✝ : Group G
        X Y : Action FintypeCat (MonCat.of G)
        f : Quiver.Hom X Y
        g : G
        y : ↑(CategoryTheory.FintypeCat.imageComplement f.hom)
        ⊢ Membership.mem (HasCompl.compl (Set.range f.hom)) (Y.ρ g ↑y)
      -/
      intro ⟨x, h⟩
      /-
        G : Type u
        inst✝ : Group G
        X Y : Action FintypeCat (MonCat.of G)
        f : Quiver.Hom X Y
        g : G
        y : ↑(CategoryTheory.FintypeCat.imageComplement f.hom)
        x : ↑X.V
        h : Eq (f.hom x) (Y.ρ g ↑y)
        ⊢ False
      -/
      apply y.property
      /-
        G : Type u
        inst✝ : Group G
        X Y : Action FintypeCat (MonCat.of G)
        f : Quiver.Hom X Y
        g : G
        y : ↑(CategoryTheory.FintypeCat.imageComplement f.hom)
        x : ↑X.V
        h : Eq (f.hom x) (Y.ρ g ↑y)
        ⊢ Membership.mem (Set.range f.hom) ↑y
      -/
      use X.ρ g⁻¹ x
      calc (X.ρ g⁻¹ ≫ f.hom) x
          = (Y.ρ g⁻¹ * Y.ρ g) y.val := by rw [f.comm, FintypeCat.comp_apply, h]; rfl
        _ = y.val := by rw [← map_mul, inv_mul_cancel, Action.ρ_one, FintypeCat.id_apply]
                   /-
                     G : Type u
                     inst✝ : Group G
                     X Y : Action FintypeCat (MonCat.of G)
                     f : Quiver.Hom X Y
                     ⊢ Eq ((fun g y => ⟨Y.ρ g ↑y, ⋯⟩) 1) 1
                   -/
    map_one' := by simp only [Action.ρ_one]; rfl
                                             /-
                                               🎉 no goals
                                             -/
    map_mul' := fun g h ↦ FintypeCat.hom_ext _ _ <| fun y ↦ Subtype.ext <| by
      /-
        G : Type u
        inst✝ : Group G
        X Y : Action FintypeCat (MonCat.of G)
        f : Quiver.Hom X Y
        g h : G
        y : ↑(CategoryTheory.FintypeCat.imageComplement f.hom)
        ⊢ Eq ↑({ toFun := fun g y => ⟨Y.ρ g ↑y, ⋯⟩, map_one' := ⋯ }.toFun (HMul.hMul g …
      -/
      exact congrFun (MonoidHom.map_mul Y.ρ g h) y.val
      /-
        🎉 no goals
      -/
  }


/-- The inclusion from the complement of the image of `f : X ⟶ Y` into `Y`. -/
def Action.imageComplementIncl {X Y : Action FintypeCat (MonCat.of G)} (f : X ⟶ Y) :
    Action.imageComplement G f ⟶ Y where
  hom := FintypeCat.imageComplementIncl f.hom
  comm _ := rfl


instance {X Y : Action FintypeCat (MonCat.of G)} (f : X ⟶ Y) :
    Mono (Action.imageComplementIncl G f) := by
  /-
    G : Type u
    inst✝ : Group G
    X Y : Action FintypeCat (MonCat.of G)
    f : Quiver.Hom X Y
    ⊢ CategoryTheory.Mono (CategoryTheory.FintypeCat.Action.imageComplementIncl G f)
  -/
  apply Functor.mono_of_mono_map (forget _)
  /-
    G : Type u
    inst✝ : Group G
    X Y : Action FintypeCat (MonCat.of G)
    f : Quiver.Hom X Y
    ⊢ CategoryTheory.Mono ((CategoryTheory.forget (Action FintypeCat (MonCat.of G) …
  -/
  apply ConcreteCategory.mono_of_injective
  /-
    case i
    G : Type u
    inst✝ : Group G
    X Y : Action FintypeCat (MonCat.of G)
    f : Quiver.Hom X Y
    ⊢ Function.Injective ⇑((CategoryTheory.forget (Action FintypeCat (MonCat.of G) …
  -/
  exact Subtype.val_injective
  /-
    🎉 no goals
  -/


/-- The category of finite sets has quotients by finite groups in arbitrary universes. -/
instance [Finite G] : HasColimitsOfShape (SingleObj G) FintypeCat.{w} := by
  /-
    G : Type u
    inst✝¹ : Group G
    inst✝ : Finite G
    ⊢ CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.SingleObj G) Fintyp …
  -/
  obtain ⟨G', hg, hf, ⟨e⟩⟩ := Finite.exists_type_univ_nonempty_mulEquiv G
  /-
    case intro.intro.intro.intro
    G : Type u
    inst✝¹ : Group G
    inst✝ : Finite G
    G' : Type ?u.12825
    hg : Group G'
    hf : Fintype G'
    e : MulEquiv G G'
    ⊢ CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.SingleObj G) Fintyp …
  -/
  exact Limits.hasColimitsOfShape_of_equivalence e.toSingleObjEquiv.symm
  /-
    🎉 no goals
  -/


noncomputable instance : PreservesFiniteLimits (forget (Action FintypeCat (MonCat.of G))) := by
  /-
    G : Type u
    inst✝ : Group G
    ⊢ CategoryTheory.Limits.PreservesFiniteLimits (CategoryTheory.forget (Action F …
  -/
  show PreservesFiniteLimits (Action.forget FintypeCat _ ⋙ FintypeCat.incl)
  /-
    G : Type u
    inst✝ : Group G
    ⊢ CategoryTheory.Limits.PreservesFiniteLimits ((Action.forget FintypeCat (MonC …
  -/
  apply comp_preservesFiniteLimits
  /-
    🎉 no goals
  -/


/-- The category of finite `G`-sets is a `PreGaloisCategory`. -/
instance : PreGaloisCategory (Action FintypeCat (MonCat.of G)) where
  hasQuotientsByFiniteGroups _ _ _ := inferInstance
  monoInducesIsoOnDirectSummand {_ _} i _ :=
    ⟨Action.imageComplement G i, Action.imageComplementIncl G i,
     ⟨isColimitOfReflects (Action.forget _ _ ⋙ FintypeCat.incl) <|
      (isColimitMapCoconeBinaryCofanEquiv (forget _) i _).symm
      (Types.isCoprodOfMono ((forget _).map i))⟩⟩


/-- The forgetful functor from finite `G`-sets to sets is a `FiberFunctor`. -/
noncomputable instance : FiberFunctor (Action.forget FintypeCat (MonCat.of G)) where
  preservesFiniteCoproducts := ⟨fun _ _ ↦ inferInstance⟩
  preservesQuotientsByFiniteGroups _ _ _ := inferInstance
  reflectsIsos := ⟨fun f (_ : IsIso f.hom) => inferInstance⟩


/-- The forgetful functor from finite `G`-sets to sets is a `FiberFunctor`. -/
noncomputable instance : FiberFunctor (forget₂ (Action FintypeCat (MonCat.of G)) FintypeCat) :=
  inferInstanceAs <| FiberFunctor (Action.forget FintypeCat (MonCat.of G))


/-- The category of finite `G`-sets is a `GaloisCategory`. -/
instance : GaloisCategory (Action FintypeCat (MonCat.of G)) where
  hasFiberFunctor := ⟨Action.forget FintypeCat (MonCat.of G), ⟨inferInstance⟩⟩


/-- The `G`-action on a connected finite `G`-set is transitive. -/
theorem Action.pretransitive_of_isConnected (X : Action FintypeCat (MonCat.of G))
    [IsConnected X] : MulAction.IsPretransitive G X.V where
  exists_smul_eq x y := by
    /- We show that the `G`-orbit of `x` is a non-initial subobject of `X` and hence by
    connectedness, the orbit equals `X.V`. -/
    /-
      G : Type u
      inst✝¹ : Group G
      X : Action FintypeCat (MonCat.of G)
      inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
      x y : ↑X.V
      ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
    -/
    let T : Set X.V := MulAction.orbit G x
    /-
      G : Type u
      inst✝¹ : Group G
      X : Action FintypeCat (MonCat.of G)
      inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
      x y : ↑X.V
      T : Set ↑X.V := MulAction.orbit G x
      ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
    -/
    have : Fintype T := Fintype.ofFinite T
    /-
      G : Type u
      inst✝¹ : Group G
      X : Action FintypeCat (MonCat.of G)
      inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
      x y : ↑X.V
      T : Set ↑X.V := MulAction.orbit G x
      this : Fintype ↑T
      ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
    -/
    letI : MulAction G (FintypeCat.of T) := inferInstanceAs <| MulAction G ↑(MulAction.orbit G x)
    /-
      G : Type u
      inst✝¹ : Group G
      X : Action FintypeCat (MonCat.of G)
      inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
      x y : ↑X.V
      T : Set ↑X.V := MulAction.orbit G x
      this✝ : Fintype ↑T
      this : MulAction G ↑(FintypeCat.of ↑T) := inferInstanceAs (MulAction G ↑(MulAc …
      ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
    -/
    let T' : Action FintypeCat (MonCat.of G) := Action.FintypeCat.ofMulAction G (FintypeCat.of T)
    /-
      G : Type u
      inst✝¹ : Group G
      X : Action FintypeCat (MonCat.of G)
      inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
      x y : ↑X.V
      T : Set ↑X.V := MulAction.orbit G x
      this✝ : Fintype ↑T
      this : MulAction G ↑(FintypeCat.of ↑T) := inferInstanceAs (MulAction G ↑(MulAc …
      T' : Action FintypeCat (MonCat.of G) := Action.FintypeCat.ofMulAction G (Finty …
      ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
    -/
    let i : T' ⟶ X := ⟨Subtype.val, fun _ ↦ rfl⟩
    /-
      G : Type u
      inst✝¹ : Group G
      X : Action FintypeCat (MonCat.of G)
      inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
      x y : ↑X.V
      T : Set ↑X.V := MulAction.orbit G x
      this✝ : Fintype ↑T
      this : MulAction G ↑(FintypeCat.of ↑T) := inferInstanceAs (MulAction G ↑(MulAc …
      T' : Action FintypeCat (MonCat.of G) := Action.FintypeCat.ofMulAction G (Finty …
      i : Quiver.Hom T' X := { hom := Subtype.val, comm := ⋯ }
      ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
    -/
    have : Mono i := ConcreteCategory.mono_of_injective _ (Subtype.val_injective)
    have : IsIso i := by
      apply IsConnected.noTrivialComponent T' i
      apply (not_initial_iff_fiber_nonempty (Action.forget _ _) T').mpr
      exact Set.Nonempty.coe_sort (MulAction.orbit_nonempty x)
    have hb : Function.Bijective i.hom := by
      apply (ConcreteCategory.isIso_iff_bijective i.hom).mp
      exact map_isIso (forget₂ _ FintypeCat) i
    /-
      G : Type u
      inst✝¹ : Group G
      X : Action FintypeCat (MonCat.of G)
      inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
      x y : ↑X.V
      T : Set ↑X.V := MulAction.orbit G x
      this✝² : Fintype ↑T
      this✝¹ : MulAction G ↑(FintypeCat.of ↑T) := inferInstanceAs (MulAction G ↑(Mul …
      T' : Action FintypeCat (MonCat.of G) := Action.FintypeCat.ofMulAction G (Finty …
      i : Quiver.Hom T' X := { hom := Subtype.val, comm := ⋯ }
      this✝ : CategoryTheory.Mono i
      this : CategoryTheory.IsIso i
      hb : Function.Bijective i.hom
      ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
    -/
    obtain ⟨⟨y', ⟨g, (hg : g • x = y')⟩⟩, (hy' : y' = y)⟩ := hb.surjective y
    /-
      case intro.mk.intro
      G : Type u
      inst✝¹ : Group G
      X : Action FintypeCat (MonCat.of G)
      inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
      x y : ↑X.V
      T : Set ↑X.V := MulAction.orbit G x
      this✝² : Fintype ↑T
      this✝¹ : MulAction G ↑(FintypeCat.of ↑T) := inferInstanceAs (MulAction G ↑(Mul …
      T' : Action FintypeCat (MonCat.of G) := Action.FintypeCat.ofMulAction G (Finty …
      i : Quiver.Hom T' X := { hom := Subtype.val, comm := ⋯ }
      this✝ : CategoryTheory.Mono i
      this : CategoryTheory.IsIso i
      hb : Function.Bijective i.hom
      y' : ↑X.V
      g : G
      hg : Eq (HSMul.hSMul g x) y'
      hy' : Eq y' y
      ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
    -/
    use g
    /-
      case h
      G : Type u
      inst✝¹ : Group G
      X : Action FintypeCat (MonCat.of G)
      inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
      x y : ↑X.V
      T : Set ↑X.V := MulAction.orbit G x
      this✝² : Fintype ↑T
      this✝¹ : MulAction G ↑(FintypeCat.of ↑T) := inferInstanceAs (MulAction G ↑(Mul …
      T' : Action FintypeCat (MonCat.of G) := Action.FintypeCat.ofMulAction G (Finty …
      i : Quiver.Hom T' X := { hom := Subtype.val, comm := ⋯ }
      this✝ : CategoryTheory.Mono i
      this : CategoryTheory.IsIso i
      hb : Function.Bijective i.hom
      y' : ↑X.V
      g : G
      hg : Eq (HSMul.hSMul g x) y'
      hy' : Eq y' y
      ⊢ Eq (HSMul.hSMul g x) y
    -/
    exact hg.trans hy'
    /-
      🎉 no goals
    -/


/-- A nonempty `G`-set with transitive `G`-action is connected. -/
theorem Action.isConnected_of_transitive (X : FintypeCat) [MulAction G X]
    [MulAction.IsPretransitive G X] [h : Nonempty X] :
    IsConnected (Action.FintypeCat.ofMulAction G X) where
  notInitial := not_initial_of_inhabited (Action.forget _ _) h.some
  noTrivialComponent Y i hm hni := by
    /- We show that the induced inclusion `i.hom` of finite sets is surjective, using the
    transitivity of the `G`-action. -/
    /-
      G : Type u
      inst✝² : Group G
      X : FintypeCat
      inst✝¹ : MulAction G ↑X
      inst✝ : MulAction.IsPretransitive G ↑X
      h : Nonempty ↑X
      Y : Action FintypeCat (MonCat.of G)
      i : Quiver.Hom Y (Action.FintypeCat.ofMulAction G X)
      hm : CategoryTheory.Mono i
      hni : CategoryTheory.Limits.IsInitial Y → False
      ⊢ CategoryTheory.IsIso i
    -/
    obtain ⟨(y : Y.V)⟩ := (not_initial_iff_fiber_nonempty (Action.forget _ _) Y).mp hni
    have : IsIso i.hom := by
      refine (ConcreteCategory.isIso_iff_bijective i.hom).mpr ⟨?_, fun x' ↦ ?_⟩
      · haveI : Mono i.hom := map_mono (forget₂ _ _) i
        exact ConcreteCategory.injective_of_mono_of_preservesPullback i.hom
      · letI x : X := i.hom y
        obtain ⟨σ, hσ⟩ := MulAction.exists_smul_eq G x x'
        use σ • y
        show (Y.ρ σ ≫ i.hom) y = x'
        rw [i.comm, FintypeCat.comp_apply]
        exact hσ
    /-
      case intro
      G : Type u
      inst✝² : Group G
      X : FintypeCat
      inst✝¹ : MulAction G ↑X
      inst✝ : MulAction.IsPretransitive G ↑X
      h : Nonempty ↑X
      Y : Action FintypeCat (MonCat.of G)
      i : Quiver.Hom Y (Action.FintypeCat.ofMulAction G X)
      hm : CategoryTheory.Mono i
      hni : CategoryTheory.Limits.IsInitial Y → False
      y : ↑Y.V
      this : CategoryTheory.IsIso i.hom
      ⊢ CategoryTheory.IsIso i
    -/
    apply isIso_of_reflects_iso i (Action.forget _ _)
    /-
      🎉 no goals
    -/


/-- A nonempty finite `G`-set is connected if and only if the `G`-action is transitive. -/
theorem Action.isConnected_iff_transitive (X : Action FintypeCat (MonCat.of G)) [Nonempty X.V] :
    IsConnected X ↔ MulAction.IsPretransitive G X.V :=
  ⟨fun _ ↦ pretransitive_of_isConnected G X, fun _ ↦ isConnected_of_transitive G X.V⟩


/-- If `X` is a connected `G`-set and `x` is an element of `X`, `X` is isomorphic
to the quotient of `G` by the stabilizer of `x` as `G`-sets. -/
noncomputable def isoQuotientStabilizerOfIsConnected (X : Action FintypeCat (MonCat.of G))
    [IsConnected X] (x : X.V) [Fintype (G ⧸ (MulAction.stabilizer G x))] :
    X ≅ G ⧸ₐ MulAction.stabilizer G x :=
  haveI : MulAction.IsPretransitive G X.V := Action.pretransitive_of_isConnected G X
  let e : X.V ≃ G ⧸ MulAction.stabilizer G x :=
    (Equiv.Set.univ X.V).symm.trans <|
      (Equiv.setCongr ((MulAction.orbit_eq_univ G x).symm)).trans <|
      MulAction.orbitEquivQuotientStabilizer G x
  Iso.symm <| Action.mkIso (FintypeCat.equivEquivIso e.symm) <| fun σ : G ↦ by
    /-
      G : Type u
      inst✝² : Group G
      X : Action FintypeCat (MonCat.of G)
      inst✝¹ : CategoryTheory.PreGaloisCategory.IsConnected X
      x : ↑X.V
      inst✝ : Fintype (HasQuotient.Quotient G (MulAction.stabilizer G x))
      this : MulAction.IsPretransitive G ↑X.V
      e : Equiv (↑X.V) (HasQuotient.Quotient G (MulAction.stabilizer G x)) := (Equiv …
      σ : G
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Action.FintypeCat.ofMulAction G (Fi …
    -/
    ext (a : G ⧸ MulAction.stabilizer G x)
    /-
      case h
      G : Type u
      inst✝² : Group G
      X : Action FintypeCat (MonCat.of G)
      inst✝¹ : CategoryTheory.PreGaloisCategory.IsConnected X
      x : ↑X.V
      inst✝ : Fintype (HasQuotient.Quotient G (MulAction.stabilizer G x))
      this : MulAction.IsPretransitive G ↑X.V
      e : Equiv (↑X.V) (HasQuotient.Quotient G (MulAction.stabilizer G x)) := (Equiv …
      σ : G
      a : HasQuotient.Quotient G (MulAction.stabilizer G x)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Action.FintypeCat.ofMulAction G (Fi …
    -/
    obtain ⟨τ, rfl⟩ := Quotient.exists_rep a
    /-
      case h.intro
      G : Type u
      inst✝² : Group G
      X : Action FintypeCat (MonCat.of G)
      inst✝¹ : CategoryTheory.PreGaloisCategory.IsConnected X
      x : ↑X.V
      inst✝ : Fintype (HasQuotient.Quotient G (MulAction.stabilizer G x))
      this : MulAction.IsPretransitive G ↑X.V
      e : Equiv (↑X.V) (HasQuotient.Quotient G (MulAction.stabilizer G x)) := (Equiv …
      σ τ : G
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Action.FintypeCat.ofMulAction G (Fi …
    -/
    exact mul_smul σ τ x
    /-
      🎉 no goals
    -/


