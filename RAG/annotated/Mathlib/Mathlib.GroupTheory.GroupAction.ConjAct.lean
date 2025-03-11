/-- A type alias for a group `G`. `ConjAct G` acts on `G` by conjugation -/
def ConjAct : Type _ :=
  G


instance [Group G] : Group (ConjAct G) := ‹Group G›


instance [DivInvMonoid G] : DivInvMonoid (ConjAct G) := ‹DivInvMonoid G›


instance [GroupWithZero G] : GroupWithZero (ConjAct G) := ‹GroupWithZero G›


instance [Fintype G] : Fintype (ConjAct G) := ‹Fintype G›


@[simp]
theorem card [Fintype G] : Fintype.card (ConjAct G) = Fintype.card G :=
  rfl


instance : Inhabited (ConjAct G) :=
  ⟨1⟩


/-- Reinterpret `g : ConjAct G` as an element of `G`. -/
def ofConjAct : ConjAct G ≃* G where
  toFun := id
  invFun := id
  left_inv := fun _ => rfl
  right_inv := fun _ => rfl
  map_mul' := fun _ _ => rfl


/-- Reinterpret `g : G` as an element of `ConjAct G`. -/
def toConjAct : G ≃* ConjAct G :=
  ofConjAct.symm


/-- A recursor for `ConjAct`, for use as `induction x` when `x : ConjAct G`. -/
@[elab_as_elim, cases_eliminator, induction_eliminator]
protected def rec {C : ConjAct G → Sort*} (h : ∀ g, C (toConjAct g)) : ∀ g, C g :=
  h


@[simp]
theorem «forall» (p : ConjAct G → Prop) : (∀ x : ConjAct G, p x) ↔ ∀ x : G, p (toConjAct x) :=
  id Iff.rfl


@[simp]
theorem of_mul_symm_eq : (@ofConjAct G _).symm = toConjAct :=
  rfl


@[simp]
theorem to_mul_symm_eq : (@toConjAct G _).symm = ofConjAct :=
  rfl


@[simp]
theorem toConjAct_ofConjAct (x : ConjAct G) : toConjAct (ofConjAct x) = x :=
  rfl


@[simp]
theorem ofConjAct_toConjAct (x : G) : ofConjAct (toConjAct x) = x :=
  rfl

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): removed `simp` attribute because `simpNF` says it can prove it

theorem ofConjAct_one : ofConjAct (1 : ConjAct G) = 1 :=
  rfl

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): removed `simp` attribute because `simpNF` says it can prove it

theorem toConjAct_one : toConjAct (1 : G) = 1 :=
  rfl


@[simp]
theorem ofConjAct_inv (x : ConjAct G) : ofConjAct x⁻¹ = (ofConjAct x)⁻¹ :=
  rfl


@[simp]
theorem toConjAct_inv (x : G) : toConjAct x⁻¹ = (toConjAct x)⁻¹ :=
  rfl

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): removed `simp` attribute because `simpNF` says it can prove it

theorem ofConjAct_mul (x y : ConjAct G) : ofConjAct (x * y) = ofConjAct x * ofConjAct y :=
  rfl

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): removed `simp` attribute because `simpNF` says it can prove it

theorem toConjAct_mul (x y : G) : toConjAct (x * y) = toConjAct x * toConjAct y :=
  rfl


instance : SMul (ConjAct G) G where smul g h := ofConjAct g * h * (ofConjAct g)⁻¹


theorem smul_def (g : ConjAct G) (h : G) : g • h = ofConjAct g * h * (ofConjAct g)⁻¹ :=
  rfl


theorem toConjAct_smul (g h : G) : toConjAct g • h = g * h * g⁻¹ :=
  rfl


instance unitsScalar : SMul (ConjAct Mˣ) M where smul g h := ofConjAct g * h * ↑(ofConjAct g)⁻¹


theorem units_smul_def (g : ConjAct Mˣ) (h : M) : g • h = ofConjAct g * h * ↑(ofConjAct g)⁻¹ :=
  rfl


instance unitsMulDistribMulAction : MulDistribMulAction (ConjAct Mˣ) M where
                 /-
                   α : Type u_1
                   M : Type u_2
                   G : Type u_3
                   G₀ : Type u_4
                   R : Type u_5
                   K : Type u_6
                   inst✝ : Monoid M
                   ⊢ ∀ (b : M), Eq (HSMul.hSMul 1 b) b
                 -/
  one_smul := by simp [units_smul_def]
                 /-
                   🎉 no goals
                 -/
                 /-
                   α : Type u_1
                   M : Type u_2
                   G : Type u_3
                   G₀ : Type u_4
                   R : Type u_5
                   K : Type u_6
                   inst✝ : Monoid M
                   ⊢ ∀ (x y : ConjAct (Units M)) (b : M), Eq (HSMul.hSMul (HMul.hMul x y) b) (HSM …
                 -/
  mul_smul := by simp [units_smul_def, mul_assoc]
                 /-
                   🎉 no goals
                 -/
                 /-
                   α : Type u_1
                   M : Type u_2
                   G : Type u_3
                   G₀ : Type u_4
                   R : Type u_5
                   K : Type u_6
                   inst✝ : Monoid M
                   ⊢ ∀ (r : ConjAct (Units M)) (x y : M), Eq (HSMul.hSMul r (HMul.hMul x y)) (HMu …
                 -/
  smul_mul := by simp [units_smul_def, mul_assoc]
                 /-
                   🎉 no goals
                 -/
                 /-
                   α : Type u_1
                   M : Type u_2
                   G : Type u_3
                   G₀ : Type u_4
                   R : Type u_5
                   K : Type u_6
                   inst✝ : Monoid M
                   ⊢ ∀ (r : ConjAct (Units M)), Eq (HSMul.hSMul r 1) 1
                 -/
  smul_one := by simp [units_smul_def]
                 /-
                   🎉 no goals
                 -/



instance unitsSMulCommClass [SMul α M] [SMulCommClass α M M] [IsScalarTower α M M] :
    SMulCommClass α (ConjAct Mˣ) M where
                         /-
                           α : Type u_1
                           M : Type u_2
                           G : Type u_3
                           G₀ : Type u_4
                           R : Type u_5
                           K : Type u_6
                           inst✝³ : Monoid M
                           inst✝² : SMul α M
                           inst✝¹ : SMulCommClass α M M
                           inst✝ : IsScalarTower α M M
                           a : α
                           um : ConjAct (Units M)
                           m : M
                           ⊢ Eq (HSMul.hSMul a (HSMul.hSMul um m)) (HSMul.hSMul um (HSMul.hSMul a m))
                         -/
  smul_comm a um m := by rw [units_smul_def, units_smul_def, mul_smul_comm, smul_mul_assoc]
                         /-
                           🎉 no goals
                         -/


instance unitsSMulCommClass' [SMul α M] [SMulCommClass M α M] [IsScalarTower α M M] :
    SMulCommClass (ConjAct Mˣ) α M :=
  haveI : SMulCommClass α M M := SMulCommClass.symm _ _ _
  SMulCommClass.symm _ _ _


instance unitsMulSemiringAction : MulSemiringAction (ConjAct Rˣ) R :=
  { ConjAct.unitsMulDistribMulAction with
                    /-
                      α : Type u_1
                      M : Type u_2
                      G : Type u_3
                      G₀ : Type u_4
                      R : Type u_5
                      K : Type u_6
                      inst✝ : Semiring R
                      ⊢ ∀ (a : ConjAct (Units R)), Eq (HSMul.hSMul a 0) 0
                    -/
    smul_zero := by simp [units_smul_def]
                    /-
                      🎉 no goals
                    -/
                   /-
                     α : Type u_1
                     M : Type u_2
                     G : Type u_3
                     G₀ : Type u_4
                     R : Type u_5
                     K : Type u_6
                     inst✝ : Semiring R
                     ⊢ ∀ (a : ConjAct (Units R)) (x y : R), Eq (HSMul.hSMul a (HAdd.hAdd x y)) (HAd …
                   -/
    smul_add := by simp [units_smul_def, mul_add, add_mul] }
                   /-
                     🎉 no goals
                   -/


theorem ofConjAct_zero : ofConjAct (0 : ConjAct G₀) = 0 :=
  rfl

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): removed `simp` attribute because `simpNF` says it can prove it

theorem toConjAct_zero : toConjAct (0 : G₀) = 0 :=
  rfl


instance mulAction₀ : MulAction (ConjAct G₀) G₀ where
                 /-
                   α : Type u_1
                   M : Type u_2
                   G : Type u_3
                   G₀ : Type u_4
                   R : Type u_5
                   K : Type u_6
                   inst✝ : GroupWithZero G₀
                   ⊢ ∀ (b : G₀), Eq (HSMul.hSMul 1 b) b
                 -/
  one_smul := by simp [smul_def]
                 /-
                   🎉 no goals
                 -/
                 /-
                   α : Type u_1
                   M : Type u_2
                   G : Type u_3
                   G₀ : Type u_4
                   R : Type u_5
                   K : Type u_6
                   inst✝ : GroupWithZero G₀
                   ⊢ ∀ (x y : ConjAct G₀) (b : G₀), Eq (HSMul.hSMul (HMul.hMul x y) b) (HSMul.hSM …
                 -/
  mul_smul := by simp [smul_def, mul_assoc]
                 /-
                   🎉 no goals
                 -/


instance smulCommClass₀ [SMul α G₀] [SMulCommClass α G₀ G₀] [IsScalarTower α G₀ G₀] :
    SMulCommClass α (ConjAct G₀) G₀ where
                         /-
                           α : Type u_1
                           M : Type u_2
                           G : Type u_3
                           G₀ : Type u_4
                           R : Type u_5
                           K : Type u_6
                           inst✝³ : GroupWithZero G₀
                           inst✝² : SMul α G₀
                           inst✝¹ : SMulCommClass α G₀ G₀
                           inst✝ : IsScalarTower α G₀ G₀
                           a : α
                           ug : ConjAct G₀
                           g : G₀
                           ⊢ Eq (HSMul.hSMul a (HSMul.hSMul ug g)) (HSMul.hSMul ug (HSMul.hSMul a g))
                         -/
  smul_comm a ug g := by rw [smul_def, smul_def, mul_smul_comm, smul_mul_assoc]
                         /-
                           🎉 no goals
                         -/


instance smulCommClass₀' [SMul α G₀] [SMulCommClass G₀ α G₀] [IsScalarTower α G₀ G₀] :
    SMulCommClass (ConjAct G₀) α G₀ :=
  haveI := SMulCommClass.symm G₀ α G₀
  SMulCommClass.symm _ _ _


instance distribMulAction₀ : DistribMulAction (ConjAct K) K :=
  { ConjAct.mulAction₀ with
                    /-
                      α : Type u_1
                      M : Type u_2
                      G : Type u_3
                      G₀ : Type u_4
                      R : Type u_5
                      K : Type u_6
                      inst✝ : DivisionRing K
                      ⊢ ∀ (a : ConjAct K), Eq (HSMul.hSMul a 0) 0
                    -/
    smul_zero := by simp [smul_def]
                    /-
                      🎉 no goals
                    -/
                   /-
                     α : Type u_1
                     M : Type u_2
                     G : Type u_3
                     G₀ : Type u_4
                     R : Type u_5
                     K : Type u_6
                     inst✝ : DivisionRing K
                     ⊢ ∀ (a : ConjAct K) (x y : K), Eq (HSMul.hSMul a (HAdd.hAdd x y)) (HAdd.hAdd ( …
                   -/
    smul_add := by simp [smul_def, mul_add, add_mul] }
                   /-
                     🎉 no goals
                   -/


instance : MulDistribMulAction (ConjAct G) G where
                 /-
                   α : Type u_1
                   M : Type u_2
                   G : Type u_3
                   G₀ : Type u_4
                   R : Type u_5
                   K : Type u_6
                   inst✝ : Group G
                   ⊢ ∀ (r : ConjAct G) (x y : G), Eq (HSMul.hSMul r (HMul.hMul x y)) (HMul.hMul ( …
                 -/
  smul_mul := by simp [smul_def]
                 /-
                   α : Type u_1
                   M : Type u_2
                   G : Type u_3
                   G₀ : Type u_4
                   R : Type u_5
                   K : Type u_6
                   inst✝ : Group G
                   ⊢ ∀ (b : G), Eq (HSMul.hSMul 1 b) b
                 -/
                 /-
                   🎉 no goals
                 -/
                 /-
                   🎉 no goals
                 -/
                 /-
                   α : Type u_1
                   M : Type u_2
                   G : Type u_3
                   G₀ : Type u_4
                   R : Type u_5
                   K : Type u_6
                   inst✝ : Group G
                   ⊢ ∀ (x y : ConjAct G) (b : G), Eq (HSMul.hSMul (HMul.hMul x y) b) (HSMul.hSMul …
                 -/
                 /-
                   α : Type u_1
                   M : Type u_2
                   G : Type u_3
                   G₀ : Type u_4
                   R : Type u_5
                   K : Type u_6
                   inst✝ : Group G
                   ⊢ ∀ (r : ConjAct G), Eq (HSMul.hSMul r 1) 1
                 -/
                 /-
                   🎉 no goals
                 -/
  smul_one := by simp [smul_def]
                 /-
                   🎉 no goals
                 -/
  one_smul := by simp [smul_def]
  mul_smul := by simp [smul_def, mul_assoc]


instance smulCommClass [SMul α G] [SMulCommClass α G G] [IsScalarTower α G G] :
    SMulCommClass α (ConjAct G) G where
                         /-
                           α : Type u_1
                           M : Type u_2
                           G : Type u_3
                           G₀ : Type u_4
                           R : Type u_5
                           K : Type u_6
                           inst✝³ : Group G
                           inst✝² : SMul α G
                           inst✝¹ : SMulCommClass α G G
                           inst✝ : IsScalarTower α G G
                           a : α
                           ug : ConjAct G
                           g : G
                           ⊢ Eq (HSMul.hSMul a (HSMul.hSMul ug g)) (HSMul.hSMul ug (HSMul.hSMul a g))
                         -/
  smul_comm a ug g := by rw [smul_def, smul_def, mul_smul_comm, smul_mul_assoc]
                         /-
                           🎉 no goals
                         -/


instance smulCommClass' [SMul α G] [SMulCommClass G α G] [IsScalarTower α G G] :
    SMulCommClass (ConjAct G) α G :=
  haveI := SMulCommClass.symm G α G
  SMulCommClass.symm _ _ _


theorem smul_eq_mulAut_conj (g : ConjAct G) (h : G) : g • h = MulAut.conj (ofConjAct g) h :=
  rfl


/-- The set of fixed points of the conjugation action of `G` on itself is the center of `G`. -/
theorem fixedPoints_eq_center : fixedPoints (ConjAct G) G = center G := by
  /-
    G : Type u_3
    inst✝ : Group G
    ⊢ Eq (MulAction.fixedPoints (ConjAct G) G) ↑(Subgroup.center G)
  -/
  ext x
  /-
    case h
    G : Type u_3
    inst✝ : Group G
    x : G
    ⊢ Iff (Membership.mem (MulAction.fixedPoints (ConjAct G) G) x) (Membership.mem …
  -/
  simp [mem_center_iff, smul_def, mul_inv_eq_iff_eq_mul]
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_orbit_conjAct {g h : G} : g ∈ orbit (ConjAct G) h ↔ IsConj g h := by
  /-
    G : Type u_3
    inst✝ : Group G
    g h : G
    ⊢ Iff (Membership.mem (MulAction.orbit (ConjAct G) h) g) (IsConj g h)
  -/
  rw [isConj_comm, isConj_iff, mem_orbit_iff]; rfl
                                               /-
                                                 🎉 no goals
                                               -/


theorem orbitRel_conjAct : ⇑(orbitRel (ConjAct G) G) = IsConj :=
                        /-
                          G : Type u_3
                          inst✝ : Group G
                          g h : G
                          ⊢ Eq ((MulAction.orbitRel (ConjAct G) G) g h) (IsConj g h)
                        -/
  funext₂ fun g h => by rw [orbitRel_apply, mem_orbit_conjAct]
                        /-
                          🎉 no goals
                        -/


theorem orbit_eq_carrier_conjClasses (g : G) :
    orbit (ConjAct G) g = (ConjClasses.mk g).carrier := by
  /-
    G : Type u_3
    inst✝ : Group G
    g : G
    ⊢ Eq (MulAction.orbit (ConjAct G) g) (ConjClasses.mk g).carrier
  -/
  ext h
  /-
    case h
    G : Type u_3
    inst✝ : Group G
    g h : G
    ⊢ Iff (Membership.mem (MulAction.orbit (ConjAct G) g) h) (Membership.mem (Conj …
  -/
  rw [ConjClasses.mem_carrier_iff_mk_eq, ConjClasses.mk_eq_mk_iff_isConj, mem_orbit_conjAct]
  /-
    🎉 no goals
  -/


theorem stabilizer_eq_centralizer (g : G) :
    stabilizer (ConjAct G) g = centralizer (zpowers (toConjAct g) : Set (ConjAct G)) :=
  le_antisymm (le_centralizer_iff.mp (zpowers_le.mpr fun _ => mul_inv_eq_iff_eq_mul.mp)) fun _ h =>
    mul_inv_eq_of_eq_mul (h g (mem_zpowers g)).symm


theorem _root_.Subgroup.centralizer_eq_comap_stabilizer (g : G) :
    Subgroup.centralizer {g} = Subgroup.comap ConjAct.toConjAct.toMonoidHom
      (MulAction.stabilizer (ConjAct G) g) := by
  /-
    G : Type u_3
    inst✝ : Group G
    g : G
    ⊢ Eq (Subgroup.centralizer (Singleton.singleton g)) (Subgroup.comap ConjAct.to …
  -/
  ext k
-- NOTE: `Subgroup.mem_centralizer_iff` should probably be stated
-- with the equality in the other direction
  /-
    case h
    G : Type u_3
    inst✝ : Group G
    g k : G
    ⊢ Iff (Membership.mem (Subgroup.centralizer (Singleton.singleton g)) k) (Membe …
  -/
  simp only [mem_centralizer_iff, Set.mem_singleton_iff, forall_eq, ConjAct.toConjAct_smul]
  /-
    case h
    G : Type u_3
    inst✝ : Group G
    g k : G
    ⊢ Iff (Eq (HMul.hMul g k) (HMul.hMul k g)) (Membership.mem (Subgroup.comap Con …
  -/
  rw [eq_comm]
  /-
    case h
    G : Type u_3
    inst✝ : Group G
    g k : G
    ⊢ Iff (Eq (HMul.hMul k g) (HMul.hMul g k)) (Membership.mem (Subgroup.comap Con …
  -/
  exact Iff.symm mul_inv_eq_iff_eq_mul
  /-
    🎉 no goals
  -/


/-- As normal subgroups are closed under conjugation, they inherit the conjugation action
  of the underlying group. -/
instance Subgroup.conjAction {H : Subgroup G} [hH : H.Normal] : SMul (ConjAct G) H :=
  ⟨fun g h => ⟨g • (h : G), hH.conj_mem h.1 h.2 (ofConjAct g)⟩⟩


theorem Subgroup.val_conj_smul {H : Subgroup G} [H.Normal] (g : ConjAct G) (h : H) :
    ↑(g • h) = g • (h : G) :=
  rfl


instance Subgroup.conjMulDistribMulAction {H : Subgroup G} [H.Normal] :
    MulDistribMulAction (ConjAct G) H :=
  Subtype.coe_injective.mulDistribMulAction H.subtype Subgroup.val_conj_smul


/-- Group conjugation on a normal subgroup. Analogous to `MulAut.conj`. -/
def _root_.MulAut.conjNormal {H : Subgroup G} [H.Normal] : G →* MulAut H :=
  (MulDistribMulAction.toMulAut (ConjAct G) H).comp toConjAct.toMonoidHom


@[simp]
theorem _root_.MulAut.conjNormal_apply {H : Subgroup G} [H.Normal] (g : G) (h : H) :
    ↑(MulAut.conjNormal g h) = g * h * g⁻¹ :=
  rfl


@[simp]
theorem _root_.MulAut.conjNormal_symm_apply {H : Subgroup G} [H.Normal] (g : G) (h : H) :
    ↑((MulAut.conjNormal g).symm h) = g⁻¹ * h * g := by
  /-
    G : Type u_3
    inst✝¹ : Group G
    H : Subgroup G
    inst✝ : H.Normal
    g : G
    h : Subtype fun x => Membership.mem H x
    ⊢ Eq (↑((MulEquiv.symm (MulAut.conjNormal g)) h)) (HMul.hMul (HMul.hMul (Inv.i …
  -/
  change _ * _⁻¹⁻¹ = _
  /-
    G : Type u_3
    inst✝¹ : Group G
    H : Subgroup G
    inst✝ : H.Normal
    g : G
    h : Subtype fun x => Membership.mem H x
    ⊢ Eq (HMul.hMul (HMul.hMul (ConjAct.ofConjAct (Inv.inv (ConjAct.toConjAct.toMo …
  -/
  rw [inv_inv]
  /-
    G : Type u_3
    inst✝¹ : Group G
    H : Subgroup G
    inst✝ : H.Normal
    g : G
    h : Subtype fun x => Membership.mem H x
    ⊢ Eq (HMul.hMul (HMul.hMul (ConjAct.ofConjAct (Inv.inv (ConjAct.toConjAct.toMo …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem _root_.MulAut.conjNormal_inv_apply {H : Subgroup G} [H.Normal] (g : G) (h : H) :
    ↑((MulAut.conjNormal g)⁻¹ h) = g⁻¹ * h * g :=
  MulAut.conjNormal_symm_apply g h


theorem _root_.MulAut.conjNormal_val {H : Subgroup G} [H.Normal] {h : H} :
    MulAut.conjNormal ↑h = MulAut.conj h :=
  MulEquiv.ext fun _ => rfl


instance normal_of_characteristic_of_normal {H : Subgroup G} [hH : H.Normal] {K : Subgroup H}
    [h : K.Characteristic] : (K.map H.subtype).Normal :=
  ⟨fun a ha b => by
    /-
      α : Type u_1
      M : Type u_2
      G : Type u_3
      G₀ : Type u_4
      R : Type u_5
      K✝ : Type u_6
      inst✝ : Group G
      H : Subgroup G
      hH : H.Normal
      K : Subgroup (Subtype fun x => Membership.mem H x)
      h : K.Characteristic
      a : G
      ha : Membership.mem (Subgroup.map H.subtype K) a
      b : G
      ⊢ Membership.mem (Subgroup.map H.subtype K) (HMul.hMul (HMul.hMul b a) (Inv.in …
    -/
    obtain ⟨a, ha, rfl⟩ := ha
    exact K.apply_coe_mem_map H.subtype
      ⟨_, (SetLike.ext_iff.mp (h.fixed (MulAut.conjNormal b)) a).mpr ha⟩⟩


/-- The stabilizer of `Mˣ` acting on itself by conjugation at `x : Mˣ` is exactly the
units of the centralizer of `x : M`. -/
@[simps! apply_coe_val symm_apply_val_coe]
def unitsCentralizerEquiv (x : Mˣ) :
    (Submonoid.centralizer ({↑x} : Set M))ˣ ≃* MulAction.stabilizer (ConjAct Mˣ) x :=
  MulEquiv.symm
  { toFun := MonoidHom.toHomUnits <|
      { toFun := fun u ↦ ⟨↑(ConjAct.ofConjAct u.1 : Mˣ), by
          /-
            α : Type u_1
            M : Type u_2
            G : Type u_3
            G₀ : Type u_4
            R : Type u_5
            K : Type u_6
            inst✝ : Monoid M
            x : Units M
            u : Subtype fun x_1 => Membership.mem (MulAction.stabilizer (ConjAct (Units M) …
            ⊢ Membership.mem (Submonoid.centralizer (Singleton.singleton ↑x)) ↑(ConjAct.of …
          -/
          rintro x ⟨rfl⟩
          /-
            case refl
            α : Type u_1
            M : Type u_2
            G : Type u_3
            G₀ : Type u_4
            R : Type u_5
            K : Type u_6
            inst✝ : Monoid M
            x : Units M
            u : Subtype fun x_1 => Membership.mem (MulAction.stabilizer (ConjAct (Units M) …
            ⊢ Eq (HMul.hMul ↑x ↑(ConjAct.ofConjAct ↑u)) (HMul.hMul ↑(ConjAct.ofConjAct ↑u) …
          -/
          have : (u : ConjAct Mˣ) • x = x := u.2
          /-
            case refl
            α : Type u_1
            M : Type u_2
            G : Type u_3
            G₀ : Type u_4
            R : Type u_5
            K : Type u_6
            inst✝ : Monoid M
            x : Units M
            u : Subtype fun x_1 => Membership.mem (MulAction.stabilizer (ConjAct (Units M) …
            this : Eq (HSMul.hSMul (↑u) x) x
            ⊢ Eq (HMul.hMul ↑x ↑(ConjAct.ofConjAct ↑u)) (HMul.hMul ↑(ConjAct.ofConjAct ↑u) …
          -/
          rwa [ConjAct.smul_def, mul_inv_eq_iff_eq_mul, Units.ext_iff, eq_comm] at this⟩,
          /-
            🎉 no goals
          -/
        map_one' := rfl,
        map_mul' := fun _ _ ↦ rfl }
    invFun := fun u ↦
      ⟨ConjAct.toConjAct (Units.map (Submonoid.centralizer ({↑x} : Set M)).subtype u), by
      /-
        α : Type u_1
        M : Type u_2
        G : Type u_3
        G₀ : Type u_4
        R : Type u_5
        K : Type u_6
        inst✝ : Monoid M
        x : Units M
        u : Units (Subtype fun x_1 => Membership.mem (Submonoid.centralizer (Singleton …
        ⊢ Membership.mem (MulAction.stabilizer (ConjAct (Units M)) x) (ConjAct.toConjA …
      -/
      change _ • _ = _
      /-
        α : Type u_1
        M : Type u_2
        G : Type u_3
        G₀ : Type u_4
        R : Type u_5
        K : Type u_6
        inst✝ : Monoid M
        x : Units M
        u : Units (Subtype fun x_1 => Membership.mem (Submonoid.centralizer (Singleton …
        ⊢ Eq (HSMul.hSMul (ConjAct.toConjAct ((Units.map (Submonoid.centralizer (Singl …
      -/
      simp only [ConjAct.smul_def, ConjAct.ofConjAct_toConjAct, mul_inv_eq_iff_eq_mul]
      /-
        α : Type u_1
        M : Type u_2
        G : Type u_3
        G₀ : Type u_4
        R : Type u_5
        K : Type u_6
        inst✝ : Monoid M
        x : Units M
        u : Units (Subtype fun x_1 => Membership.mem (Submonoid.centralizer (Singleton …
        ⊢ Eq (HMul.hMul ((Units.map (Submonoid.centralizer (Singleton.singleton ↑x)).s …
      -/
      exact Units.ext <| (u.1.2 x <| Set.mem_singleton _).symm⟩
      /-
        🎉 no goals
      -/
                           /-
                             α : Type u_1
                             M : Type u_2
                             G : Type u_3
                             G₀ : Type u_4
                             R : Type u_5
                             K : Type u_6
                             inst✝ : Monoid M
                             x : Units M
                             x✝ : Subtype fun x_1 => Membership.mem (MulAction.stabilizer (ConjAct (Units M …
                             ⊢ Eq ((fun u => ⟨ConjAct.toConjAct ((Units.map (Submonoid.centralizer (Singlet …
                           -/
    left_inv := fun _ ↦ by ext; rfl
                                /-
                                  🎉 no goals
                                -/
                            /-
                              α : Type u_1
                              M : Type u_2
                              G : Type u_3
                              G₀ : Type u_4
                              R : Type u_5
                              K : Type u_6
                              inst✝ : Monoid M
                              x : Units M
                              x✝ : Units (Subtype fun x_1 => Membership.mem (Submonoid.centralizer (Singleto …
                              ⊢ Eq ({ toFun := fun u => ⟨↑(ConjAct.ofConjAct ↑u), ⋯⟩, map_one' := ?m.67145,  …
                            -/
    right_inv := fun _ ↦ by ext; rfl
                                 /-
                                   🎉 no goals
                                 -/
    map_mul' := map_mul _ }


