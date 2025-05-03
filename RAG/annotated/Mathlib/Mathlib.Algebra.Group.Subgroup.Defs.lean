/-- `InvMemClass S G` states `S` is a type of subsets `s ⊆ G` closed under inverses. -/
class InvMemClass (S : Type*) (G : outParam Type*) [Inv G] [SetLike S G] : Prop where
  /-- `s` is closed under inverses -/
  inv_mem : ∀ {s : S} {x}, x ∈ s → x⁻¹ ∈ s


/-- `NegMemClass S G` states `S` is a type of subsets `s ⊆ G` closed under negation. -/
class NegMemClass (S : Type*) (G : outParam Type*) [Neg G] [SetLike S G] : Prop where
  /-- `s` is closed under negation -/
  neg_mem : ∀ {s : S} {x}, x ∈ s → -x ∈ s


/-- `SubgroupClass S G` states `S` is a type of subsets `s ⊆ G` that are subgroups of `G`. -/
class SubgroupClass (S : Type*) (G : outParam Type*) [DivInvMonoid G] [SetLike S G]
    extends SubmonoidClass S G, InvMemClass S G : Prop


/-- `AddSubgroupClass S G` states `S` is a type of subsets `s ⊆ G` that are
additive subgroups of `G`. -/
class AddSubgroupClass (S : Type*) (G : outParam Type*) [SubNegMonoid G] [SetLike S G]
    extends AddSubmonoidClass S G, NegMemClass S G : Prop


@[to_additive (attr := simp)]
theorem inv_mem_iff {S G} [InvolutiveInv G] {_ : SetLike S G} [InvMemClass S G] {H : S}
    {x : G} : x⁻¹ ∈ H ↔ x ∈ H :=
  ⟨fun h => inv_inv x ▸ inv_mem h, inv_mem⟩


/-- A subgroup is closed under division. -/
@[to_additive (attr := aesop safe apply (rule_sets := [SetLike]))
  "An additive subgroup is closed under subtraction."]
theorem div_mem {x y : M} (hx : x ∈ H) (hy : y ∈ H) : x / y ∈ H := by
  /-
    M : Type u_3
    S : Type u_4
    inst✝¹ : DivInvMonoid M
    inst✝ : SetLike S M
    hSM : SubgroupClass S M
    H : S
    x y : M
    hx : Membership.mem H x
    hy : Membership.mem H y
    ⊢ Membership.mem H (HDiv.hDiv x y)
  -/
  rw [div_eq_mul_inv]; exact mul_mem hx (inv_mem hy)
                       /-
                         🎉 no goals
                       -/


@[to_additive (attr := aesop safe apply (rule_sets := [SetLike]))]
theorem zpow_mem {x : M} (hx : x ∈ K) : ∀ n : ℤ, x ^ n ∈ K
  | (n : ℕ) => by
    /-
      M : Type u_3
      S : Type u_4
      inst✝¹ : DivInvMonoid M
      inst✝ : SetLike S M
      hSM : SubgroupClass S M
      K : S
      x : M
      hx : Membership.mem K x
      n : Nat
      ⊢ Membership.mem K (HPow.hPow x ↑n)
    -/
    rw [zpow_natCast]
    /-
      M : Type u_3
      S : Type u_4
      inst✝¹ : DivInvMonoid M
      inst✝ : SetLike S M
      hSM : SubgroupClass S M
      K : S
      x : M
      hx : Membership.mem K x
      n : Nat
      ⊢ Membership.mem K (HPow.hPow x n)
    -/
    exact pow_mem hx n
    /-
      🎉 no goals
    -/
  | -[n+1] => by
    /-
      M : Type u_3
      S : Type u_4
      inst✝¹ : DivInvMonoid M
      inst✝ : SetLike S M
      hSM : SubgroupClass S M
      K : S
      x : M
      hx : Membership.mem K x
      n : Nat
      ⊢ Membership.mem K (HPow.hPow x (Int.negSucc n))
    -/
    rw [zpow_negSucc]
    /-
      M : Type u_3
      S : Type u_4
      inst✝¹ : DivInvMonoid M
      inst✝ : SetLike S M
      hSM : SubgroupClass S M
      K : S
      x : M
      hx : Membership.mem K x
      n : Nat
      ⊢ Membership.mem K (Inv.inv (HPow.hPow x (HAdd.hAdd n 1)))
    -/
    exact inv_mem (pow_mem hx n.succ)
    /-
      🎉 no goals
    -/


@[to_additive /-(attr := simp)-/] -- Porting note: `simp` cannot simplify LHS
theorem exists_inv_mem_iff_exists_mem {P : G → Prop} :
    (∃ x : G, x ∈ H ∧ P x⁻¹) ↔ ∃ x ∈ H, P x := by
  /-
    G : Type u_1
    inst✝² : Group G
    S : Type u_4
    H : S
    inst✝¹ : SetLike S G
    inst✝ : SubgroupClass S G
    P : G → Prop
    ⊢ Iff (Exists fun x => And (Membership.mem H x) (P (Inv.inv x))) (Exists fun x …
  -/
  constructor <;>
      /-
        case mp
        G : Type u_1
        inst✝² : Group G
        S : Type u_4
        H : S
        inst✝¹ : SetLike S G
        inst✝ : SubgroupClass S G
        P : G → Prop
        ⊢ (Exists fun x => And (Membership.mem H x) (P (Inv.inv x))) → Exists fun x => …
      -/
      /-
        case mp.intro.intro
        G : Type u_1
        inst✝² : Group G
        S : Type u_4
        H : S
        inst✝¹ : SetLike S G
        inst✝ : SubgroupClass S G
        P : G → Prop
        x : G
        x_in : Membership.mem H x
        hx : P (Inv.inv x)
        ⊢ Exists fun x => And (Membership.mem H x) (P x)
      -/
      /-
        🎉 no goals
      -/
      /-
        case mpr.intro.intro
        G : Type u_1
        inst✝² : Group G
        S : Type u_4
        H : S
        inst✝¹ : SetLike S G
        inst✝ : SubgroupClass S G
        P : G → Prop
        x : G
        x_in : Membership.mem H x
        hx : P x
        ⊢ Exists fun x => And (Membership.mem H x) (P (Inv.inv x))
      -/
      exact ⟨x⁻¹, inv_mem x_in, by simp [hx]⟩
      /-
        🎉 no goals
      -/


@[to_additive]
theorem mul_mem_cancel_right {x y : G} (h : x ∈ H) : y * x ∈ H ↔ y ∈ H :=
                 /-
                   G : Type u_1
                   inst✝² : Group G
                   S : Type u_4
                   H : S
                   inst✝¹ : SetLike S G
                   inst✝ : SubgroupClass S G
                   x y : G
                   h : Membership.mem H x
                   hba : Membership.mem H (HMul.hMul y x)
                   ⊢ Membership.mem H y
                 -/
  ⟨fun hba => by simpa using mul_mem hba (inv_mem h), fun hb => mul_mem hb h⟩
                 /-
                   🎉 no goals
                 -/


@[to_additive]
theorem mul_mem_cancel_left {x y : G} (h : x ∈ H) : x * y ∈ H ↔ y ∈ H :=
                 /-
                   G : Type u_1
                   inst✝² : Group G
                   S : Type u_4
                   H : S
                   inst✝¹ : SetLike S G
                   inst✝ : SubgroupClass S G
                   x y : G
                   h : Membership.mem H x
                   hab : Membership.mem H (HMul.hMul x y)
                   ⊢ Membership.mem H y
                 -/
  ⟨fun hab => by simpa using mul_mem (inv_mem h) hab, mul_mem h⟩
                 /-
                   🎉 no goals
                 -/


/-- A subgroup of a group inherits an inverse. -/
@[to_additive "An additive subgroup of an `AddGroup` inherits an inverse."]
instance inv {G : Type u_1} {S : Type u_2} [Inv G] [SetLike S G]
  [InvMemClass S G] {H : S} : Inv H :=
  ⟨fun a => ⟨a⁻¹, inv_mem a.2⟩⟩


@[to_additive (attr := simp, norm_cast)]
theorem coe_inv (x : H) : (x⁻¹).1 = x.1⁻¹ :=
  rfl


@[to_additive]
theorem subset_union {H K L : S} : (H : Set G) ⊆ K ∪ L ↔ H ≤ K ∨ H ≤ L := by
  /-
    G : Type u_1
    inst✝² : Group G
    S : Type u_4
    inst✝¹ : SetLike S G
    inst✝ : SubgroupClass S G
    H K L : S
    ⊢ Iff (HasSubset.Subset (↑H) (Union.union ↑K ↑L)) (Or (LE.le H K) (LE.le H L))
  -/
  refine ⟨fun h ↦ ?_, fun h x xH ↦ h.imp (· xH) (· xH)⟩
  /-
    G : Type u_1
    inst✝² : Group G
    S : Type u_4
    inst✝¹ : SetLike S G
    inst✝ : SubgroupClass S G
    H K L : S
    h : HasSubset.Subset (↑H) (Union.union ↑K ↑L)
    ⊢ Or (LE.le H K) (LE.le H L)
  -/
  rw [or_iff_not_imp_left, SetLike.not_le_iff_exists]
  exact fun ⟨x, xH, xK⟩ y yH ↦ (h <| mul_mem xH yH).elim
    ((h yH).resolve_left fun yK ↦ xK <| (mul_mem_cancel_right yK).mp ·)
    (mul_mem_cancel_left <| (h xH).resolve_left xK).mp


/-- A subgroup of a group inherits a division -/
@[to_additive "An additive subgroup of an `AddGroup` inherits a subtraction."]
instance div {G : Type u_1} {S : Type u_2} [DivInvMonoid G] [SetLike S G]
  [SubgroupClass S G] {H : S} : Div H :=
  ⟨fun a b => ⟨a / b, div_mem a.2 b.2⟩⟩


/-- An additive subgroup of an `AddGroup` inherits an integer scaling. -/
instance _root_.AddSubgroupClass.zsmul {M S} [SubNegMonoid M] [SetLike S M]
    [AddSubgroupClass S M] {H : S} : SMul ℤ H :=
  ⟨fun n a => ⟨n • a.1, zsmul_mem a.2 n⟩⟩


/-- A subgroup of a group inherits an integer power. -/
@[to_additive existing]
instance zpow {M S} [DivInvMonoid M] [SetLike S M] [SubgroupClass S M] {H : S} : Pow H ℤ :=
  ⟨fun a n => ⟨a.1 ^ n, zpow_mem a.2 n⟩⟩


@[to_additive (attr := simp, norm_cast)]
theorem coe_div (x y : H) : (x / y).1 = x.1 / y.1 :=
  rfl


/-- A subgroup of a group inherits a group structure. -/
@[to_additive "An additive subgroup of an `AddGroup` inherits an `AddGroup` structure."]
instance (priority := 75) toGroup : Group H :=
  Subtype.coe_injective.group _ rfl (fun _ _ => rfl) (fun _ => rfl) (fun _ _ => rfl)
    (fun _ _ => rfl) fun _ _ => rfl

-- Prefer subclasses of `CommGroup` over subclasses of `SubgroupClass`.

/-- A subgroup of a `CommGroup` is a `CommGroup`. -/
@[to_additive "An additive subgroup of an `AddCommGroup` is an `AddCommGroup`."]
instance (priority := 75) toCommGroup {G : Type*} [CommGroup G] [SetLike S G] [SubgroupClass S G] :
    CommGroup H :=
  Subtype.coe_injective.commGroup _ rfl (fun _ _ => rfl) (fun _ => rfl) (fun _ _ => rfl)
    (fun _ _ => rfl) fun _ _ => rfl


/-- The natural group hom from a subgroup of group `G` to `G`. -/
@[to_additive (attr := coe)
  "The natural group hom from an additive subgroup of `AddGroup` `G` to `G`."]
protected def subtype : H →* G where
  toFun := ((↑) : H → G); map_one' := rfl; map_mul' := fun _ _ => rfl


@[to_additive (attr := simp)]
theorem coeSubtype : (SubgroupClass.subtype H : H → G) = ((↑) : H → G) := by
  /-
    G : Type u_1
    inst✝² : Group G
    S : Type u_4
    H : S
    inst✝¹ : SetLike S G
    inst✝ : SubgroupClass S G
    ⊢ Eq (⇑↑H) Subtype.val
  -/
  rfl
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp, norm_cast)]
theorem coe_pow (x : H) (n : ℕ) : ((x ^ n : H) : G) = (x : G) ^ n :=
  rfl


@[to_additive (attr := simp, norm_cast)]
theorem coe_zpow (x : H) (n : ℤ) : ((x ^ n : H) : G) = (x : G) ^ n :=
  rfl


/-- The inclusion homomorphism from a subgroup `H` contained in `K` to `K`. -/
@[to_additive "The inclusion homomorphism from an additive subgroup `H` contained in `K` to `K`."]
def inclusion {H K : S} (h : H ≤ K) : H →* K :=
  MonoidHom.mk' (fun x => ⟨x, h x.prop⟩) fun _ _=> rfl


@[to_additive (attr := simp)]
theorem inclusion_self (x : H) : inclusion le_rfl x = x := by
  /-
    G : Type u_1
    inst✝² : Group G
    S : Type u_4
    H : S
    inst✝¹ : SetLike S G
    inst✝ : SubgroupClass S G
    x : Subtype fun x => Membership.mem H x
    ⊢ Eq ((SubgroupClass.inclusion ⋯) x) x
  -/
  cases x
  /-
    case mk
    G : Type u_1
    inst✝² : Group G
    S : Type u_4
    H : S
    inst✝¹ : SetLike S G
    inst✝ : SubgroupClass S G
    val✝ : G
    property✝ : Membership.mem H val✝
    ⊢ Eq ((SubgroupClass.inclusion ⋯) ⟨val✝, property✝⟩) ⟨val✝, property✝⟩
  -/
  rfl
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem inclusion_mk {h : H ≤ K} (x : G) (hx : x ∈ H) : inclusion h ⟨x, hx⟩ = ⟨x, h hx⟩ :=
  rfl


@[to_additive]
theorem inclusion_right (h : H ≤ K) (x : K) (hx : (x : G) ∈ H) : inclusion h ⟨x, hx⟩ = x := by
  /-
    G : Type u_1
    inst✝² : Group G
    S : Type u_4
    H K : S
    inst✝¹ : SetLike S G
    inst✝ : SubgroupClass S G
    h : LE.le H K
    x : Subtype fun x => Membership.mem K x
    hx : Membership.mem H ↑x
    ⊢ Eq ((SubgroupClass.inclusion h) ⟨↑x, hx⟩) x
  -/
  cases x
  /-
    case mk
    G : Type u_1
    inst✝² : Group G
    S : Type u_4
    H K : S
    inst✝¹ : SetLike S G
    inst✝ : SubgroupClass S G
    h : LE.le H K
    val✝ : G
    property✝ : Membership.mem K val✝
    hx : Membership.mem H ↑⟨val✝, property✝⟩
    ⊢ Eq ((SubgroupClass.inclusion h) ⟨↑⟨val✝, property✝⟩, hx⟩) ⟨val✝, property✝⟩
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem inclusion_inclusion {L : S} (hHK : H ≤ K) (hKL : K ≤ L) (x : H) :
    inclusion hKL (inclusion hHK x) = inclusion (hHK.trans hKL) x := by
  /-
    G : Type u_1
    inst✝² : Group G
    S : Type u_4
    H K : S
    inst✝¹ : SetLike S G
    inst✝ : SubgroupClass S G
    L : S
    hHK : LE.le H K
    hKL : LE.le K L
    x : Subtype fun x => Membership.mem H x
    ⊢ Eq ((SubgroupClass.inclusion hKL) ((SubgroupClass.inclusion hHK) x)) ((Subgr …
  -/
  cases x
  /-
    case mk
    G : Type u_1
    inst✝² : Group G
    S : Type u_4
    H K : S
    inst✝¹ : SetLike S G
    inst✝ : SubgroupClass S G
    L : S
    hHK : LE.le H K
    hKL : LE.le K L
    val✝ : G
    property✝ : Membership.mem H val✝
    ⊢ Eq ((SubgroupClass.inclusion hKL) ((SubgroupClass.inclusion hHK) ⟨val✝, prop …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem coe_inclusion {H K : S} {h : H ≤ K} (a : H) : (inclusion h a : G) = a := by
  /-
    G : Type u_1
    inst✝² : Group G
    S : Type u_4
    inst✝¹ : SetLike S G
    inst✝ : SubgroupClass S G
    H K : S
    h : LE.le H K
    a : Subtype fun x => Membership.mem H x
    ⊢ Eq ↑((SubgroupClass.inclusion h) a) ↑a
  -/
  cases a
  /-
    case mk
    G : Type u_1
    inst✝² : Group G
    S : Type u_4
    inst✝¹ : SetLike S G
    inst✝ : SubgroupClass S G
    H K : S
    h : LE.le H K
    val✝ : G
    property✝ : Membership.mem H val✝
    ⊢ Eq ↑((SubgroupClass.inclusion h) ⟨val✝, property✝⟩) ↑⟨val✝, property✝⟩
  -/
  simp only [inclusion, MonoidHom.mk'_apply]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem subtype_comp_inclusion {H K : S} (hH : H ≤ K) :
    (SubgroupClass.subtype K).comp (inclusion hH) = SubgroupClass.subtype H := by
  /-
    G : Type u_1
    inst✝² : Group G
    S : Type u_4
    inst✝¹ : SetLike S G
    inst✝ : SubgroupClass S G
    H K : S
    hH : LE.le H K
    ⊢ Eq ((↑K).comp (SubgroupClass.inclusion hH)) ↑H
  -/
  ext
  /-
    case h
    G : Type u_1
    inst✝² : Group G
    S : Type u_4
    inst✝¹ : SetLike S G
    inst✝ : SubgroupClass S G
    H K : S
    hH : LE.le H K
    x✝ : Subtype fun x => Membership.mem H x
    ⊢ Eq (((↑K).comp (SubgroupClass.inclusion hH)) x✝) (↑H x✝)
  -/
  simp only [MonoidHom.comp_apply, coeSubtype, coe_inclusion]
  /-
    🎉 no goals
  -/


/-- A subgroup of a group `G` is a subset containing 1, closed under multiplication
and closed under multiplicative inverse. -/
structure Subgroup (G : Type*) [Group G] extends Submonoid G where
  /-- `G` is closed under inverses -/
  inv_mem' {x} : x ∈ carrier → x⁻¹ ∈ carrier


/-- An additive subgroup of an additive group `G` is a subset containing 0, closed
under addition and additive inverse. -/
structure AddSubgroup (G : Type*) [AddGroup G] extends AddSubmonoid G where
  /-- `G` is closed under negation -/
  neg_mem' {x} : x ∈ carrier → -x ∈ carrier


@[to_additive]
instance : SetLike (Subgroup G) G where
  coe s := s.carrier
  coe_injective' p q h := by
    /-
      G : Type u_1
      inst✝¹ : Group G
      A : Type u_2
      inst✝ : AddGroup A
      p q : Subgroup G
      h : Eq ((fun s => s.carrier) p) ((fun s => s.carrier) q)
      ⊢ Eq p q
    -/
    obtain ⟨⟨⟨hp,_⟩,_⟩,_⟩ := p
    /-
      case mk.mk.mk
      G : Type u_1
      inst✝¹ : Group G
      A : Type u_2
      inst✝ : AddGroup A
      q : Subgroup G
      hp : Set G
      mul_mem'✝ : ∀ {a b : G}, Membership.mem hp a → Membership.mem hp b → Membershi …
      one_mem'✝ : Membership.mem { carrier := hp, mul_mem' := mul_mem'✝ }.carrier 1
      inv_mem'✝ : ∀ {x : G}, Membership.mem { carrier := hp, mul_mem' := mul_mem'✝,  …
      h : Eq ((fun s => s.carrier) { carrier := hp, mul_mem' := mul_mem'✝, one_mem'  …
      ⊢ Eq { carrier := hp, mul_mem' := mul_mem'✝, one_mem' := one_mem'✝, inv_mem' : …
    -/
    obtain ⟨⟨⟨hq,_⟩,_⟩,_⟩ := q
    /-
      case mk.mk.mk.mk.mk.mk
      G : Type u_1
      inst✝¹ : Group G
      A : Type u_2
      inst✝ : AddGroup A
      hp : Set G
      mul_mem'✝¹ : ∀ {a b : G}, Membership.mem hp a → Membership.mem hp b → Membersh …
      one_mem'✝¹ : Membership.mem { carrier := hp, mul_mem' := mul_mem'✝¹ }.carrier 1
      inv_mem'✝¹ : ∀ {x : G}, Membership.mem { carrier := hp, mul_mem' := mul_mem'✝¹ …
      hq : Set G
      mul_mem'✝ : ∀ {a b : G}, Membership.mem hq a → Membership.mem hq b → Membershi …
      one_mem'✝ : Membership.mem { carrier := hq, mul_mem' := mul_mem'✝ }.carrier 1
      inv_mem'✝ : ∀ {x : G}, Membership.mem { carrier := hq, mul_mem' := mul_mem'✝,  …
      h : Eq ((fun s => s.carrier) { carrier := hp, mul_mem' := mul_mem'✝¹, one_mem' …
      ⊢ Eq { carrier := hp, mul_mem' := mul_mem'✝¹, one_mem' := one_mem'✝¹, inv_mem' …
    -/
    congr
    /-
      🎉 no goals
    -/

-- Porting note: Below can probably be written more uniformly

@[to_additive]
instance : SubgroupClass (Subgroup G) G where
  inv_mem := Subgroup.inv_mem' _
  one_mem _ := (Subgroup.toSubmonoid _).one_mem'
  mul_mem := (Subgroup.toSubmonoid _).mul_mem'

-- This is not a simp lemma,
-- because the simp normal form left-hand side is given by `mem_toSubmonoid` below.

@[to_additive]
theorem mem_carrier {s : Subgroup G} {x : G} : x ∈ s.carrier ↔ x ∈ s :=
  Iff.rfl


@[to_additive (attr := simp)]
theorem mem_mk {s : Set G} {x : G} (h_one) (h_mul) (h_inv) :
    x ∈ mk ⟨⟨s, h_one⟩, h_mul⟩ h_inv ↔ x ∈ s :=
  Iff.rfl


@[to_additive (attr := simp, norm_cast)]
theorem coe_set_mk {s : Set G} (h_one) (h_mul) (h_inv) :
    (mk ⟨⟨s, h_one⟩, h_mul⟩ h_inv : Set G) = s :=
  rfl


@[to_additive (attr := simp)]
theorem mk_le_mk {s t : Set G} (h_one) (h_mul) (h_inv) (h_one') (h_mul') (h_inv') :
    mk ⟨⟨s, h_one⟩, h_mul⟩ h_inv ≤ mk ⟨⟨t, h_one'⟩, h_mul'⟩ h_inv' ↔ s ⊆ t :=
  Iff.rfl


@[to_additive (attr := simp)]
theorem coe_toSubmonoid (K : Subgroup G) : (K.toSubmonoid : Set G) = K :=
  rfl


@[to_additive (attr := simp)]
theorem mem_toSubmonoid (K : Subgroup G) (x : G) : x ∈ K.toSubmonoid ↔ x ∈ K :=
  Iff.rfl


@[to_additive]
theorem toSubmonoid_injective : Function.Injective (toSubmonoid : Subgroup G → Submonoid G) :=
  -- fun p q h => SetLike.ext'_iff.2 (show _ from SetLike.ext'_iff.1 h)
  fun p q h => by
    /-
      G : Type u_1
      inst✝ : Group G
      p q : Subgroup G
      h : Eq p.toSubmonoid q.toSubmonoid
      ⊢ Eq p q
    -/
    have := SetLike.ext'_iff.1 h
    /-
      G : Type u_1
      inst✝ : Group G
      p q : Subgroup G
      h : Eq p.toSubmonoid q.toSubmonoid
      this : Eq ↑p.toSubmonoid ↑q.toSubmonoid
      ⊢ Eq p q
    -/
    rw [coe_toSubmonoid, coe_toSubmonoid] at this
    /-
      G : Type u_1
      inst✝ : Group G
      p q : Subgroup G
      h : Eq p.toSubmonoid q.toSubmonoid
      this : Eq ↑p ↑q
      ⊢ Eq p q
    -/
    exact SetLike.ext'_iff.2 this
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp)]
theorem toSubmonoid_inj {p q : Subgroup G} : p.toSubmonoid = q.toSubmonoid ↔ p = q :=
  toSubmonoid_injective.eq_iff


@[to_additive, deprecated (since := "2024-12-29")] alias toSubmonoid_eq := toSubmonoid_inj


@[to_additive (attr := mono)]
theorem toSubmonoid_strictMono : StrictMono (toSubmonoid : Subgroup G → Submonoid G) := fun _ _ =>
  id


@[to_additive (attr := mono)]
theorem toSubmonoid_mono : Monotone (toSubmonoid : Subgroup G → Submonoid G) :=
  toSubmonoid_strictMono.monotone


@[to_additive (attr := simp)]
theorem toSubmonoid_le {p q : Subgroup G} : p.toSubmonoid ≤ q.toSubmonoid ↔ p ≤ q :=
  Iff.rfl


@[to_additive (attr := simp)]
lemma coe_nonempty (s : Subgroup G) : (s : Set G).Nonempty := ⟨1, one_mem _⟩


/-- Copy of a subgroup with a new `carrier` equal to the old one. Useful to fix definitional
equalities. -/
@[to_additive
      "Copy of an additive subgroup with a new `carrier` equal to the old one.
      Useful to fix definitional equalities"]
protected def copy (K : Subgroup G) (s : Set G) (hs : s = K) : Subgroup G where
  carrier := s
  one_mem' := hs.symm ▸ K.one_mem'
  mul_mem' := hs.symm ▸ K.mul_mem'
                    /-
                      G : Type u_1
                      inst✝¹ : Group G
                      A : Type u_2
                      inst✝ : AddGroup A
                      H K✝ K : Subgroup G
                      s : Set G
                      hs : Eq s ↑K
                      x✝ : G
                      hx : Membership.mem { carrier := s, mul_mem' := ⋯, one_mem' := ⋯ }.carrier x✝
                      ⊢ Membership.mem { carrier := s, mul_mem' := ⋯, one_mem' := ⋯ }.carrier (Inv.i …
                    -/
  inv_mem' hx := by simpa [hs] using hx -- Porting note: `▸` didn't work here
                    /-
                      🎉 no goals
                    -/


@[to_additive (attr := simp)]
theorem coe_copy (K : Subgroup G) (s : Set G) (hs : s = ↑K) : (K.copy s hs : Set G) = s :=
  rfl


@[to_additive]
theorem copy_eq (K : Subgroup G) (s : Set G) (hs : s = ↑K) : K.copy s hs = K :=
  SetLike.coe_injective hs


/-- Two subgroups are equal if they have the same elements. -/
@[to_additive (attr := ext) "Two `AddSubgroup`s are equal if they have the same elements."]
theorem ext {H K : Subgroup G} (h : ∀ x, x ∈ H ↔ x ∈ K) : H = K :=
  SetLike.ext h


/-- A subgroup contains the group's 1. -/
@[to_additive "An `AddSubgroup` contains the group's 0."]
protected theorem one_mem : (1 : G) ∈ H :=
  one_mem _


/-- A subgroup is closed under multiplication. -/
@[to_additive "An `AddSubgroup` is closed under addition."]
protected theorem mul_mem {x y : G} : x ∈ H → y ∈ H → x * y ∈ H :=
  mul_mem


/-- A subgroup is closed under inverse. -/
@[to_additive "An `AddSubgroup` is closed under inverse."]
protected theorem inv_mem {x : G} : x ∈ H → x⁻¹ ∈ H :=
  inv_mem


/-- A subgroup is closed under division. -/
@[to_additive "An `AddSubgroup` is closed under subtraction."]
protected theorem div_mem {x y : G} (hx : x ∈ H) (hy : y ∈ H) : x / y ∈ H :=
  div_mem hx hy


@[to_additive]
protected theorem inv_mem_iff {x : G} : x⁻¹ ∈ H ↔ x ∈ H :=
  inv_mem_iff


@[to_additive]
protected theorem exists_inv_mem_iff_exists_mem (K : Subgroup G) {P : G → Prop} :
    (∃ x : G, x ∈ K ∧ P x⁻¹) ↔ ∃ x ∈ K, P x :=
  exists_inv_mem_iff_exists_mem


@[to_additive]
protected theorem mul_mem_cancel_right {x y : G} (h : x ∈ H) : y * x ∈ H ↔ y ∈ H :=
  mul_mem_cancel_right h


@[to_additive]
protected theorem mul_mem_cancel_left {x y : G} (h : x ∈ H) : x * y ∈ H ↔ y ∈ H :=
  mul_mem_cancel_left h


@[to_additive]
protected theorem pow_mem {x : G} (hx : x ∈ K) : ∀ n : ℕ, x ^ n ∈ K :=
  pow_mem hx


@[to_additive]
protected theorem zpow_mem {x : G} (hx : x ∈ K) : ∀ n : ℤ, x ^ n ∈ K :=
  zpow_mem hx


/-- Construct a subgroup from a nonempty set that is closed under division. -/
@[to_additive "Construct a subgroup from a nonempty set that is closed under subtraction"]
def ofDiv (s : Set G) (hsn : s.Nonempty) (hs : ∀ᵉ (x ∈ s) (y ∈ s), x * y⁻¹ ∈ s) :
    Subgroup G :=
  have one_mem : (1 : G) ∈ s := by
    /-
      G : Type u_1
      inst✝¹ : Group G
      A : Type u_2
      inst✝ : AddGroup A
      H K : Subgroup G
      s : Set G
      hsn : s.Nonempty
      hs : ∀ (x : G), Membership.mem s x → ∀ (y : G), Membership.mem s y → Membershi …
      ⊢ Membership.mem s 1
    -/
    let ⟨x, hx⟩ := hsn
    /-
      G : Type u_1
      inst✝¹ : Group G
      A : Type u_2
      inst✝ : AddGroup A
      H K : Subgroup G
      s : Set G
      hsn : s.Nonempty
      hs : ∀ (x : G), Membership.mem s x → ∀ (y : G), Membership.mem s y → Membershi …
      x : G
      hx : Membership.mem s x
      ⊢ Membership.mem s 1
    -/
    simpa using hs x hx x hx
    /-
      🎉 no goals
    -/
                                                        /-
                                                          G : Type u_1
                                                          inst✝¹ : Group G
                                                          A : Type u_2
                                                          inst✝ : AddGroup A
                                                          H K : Subgroup G
                                                          s : Set G
                                                          hsn : s.Nonempty
                                                          hs : ∀ (x : G), Membership.mem s x → ∀ (y : G), Membership.mem s y → Membershi …
                                                          one_mem : Membership.mem s 1
                                                          x : G
                                                          hx : Membership.mem s x
                                                          ⊢ Membership.mem s (Inv.inv x)
                                                        -/
  have inv_mem : ∀ x, x ∈ s → x⁻¹ ∈ s := fun x hx => by simpa using hs 1 one_mem x hx
                                                        /-
                                                          🎉 no goals
                                                        -/
  { carrier := s
    one_mem' := one_mem
    inv_mem' := inv_mem _
                                /-
                                  G : Type u_1
                                  inst✝¹ : Group G
                                  A : Type u_2
                                  inst✝ : AddGroup A
                                  H K : Subgroup G
                                  s : Set G
                                  hsn : s.Nonempty
                                  hs : ∀ (x : G), Membership.mem s x → ∀ (y : G), Membership.mem s y → Membershi …
                                  one_mem : Membership.mem s 1
                                  inv_mem : ∀ (x : G), Membership.mem s x → Membership.mem s (Inv.inv x)
                                  a✝ b✝ : G
                                  hx : Membership.mem s a✝
                                  hy : Membership.mem s b✝
                                  ⊢ Membership.mem s (HMul.hMul a✝ b✝)
                                -/
    mul_mem' := fun hx hy => by simpa using hs _ hx _ (inv_mem _ hy) }
                                /-
                                  🎉 no goals
                                -/


/-- A subgroup of a group inherits a multiplication. -/
@[to_additive "An `AddSubgroup` of an `AddGroup` inherits an addition."]
instance mul : Mul H :=
  H.toSubmonoid.mul


/-- A subgroup of a group inherits a 1. -/
@[to_additive "An `AddSubgroup` of an `AddGroup` inherits a zero."]
instance one : One H :=
  H.toSubmonoid.one


/-- A subgroup of a group inherits an inverse. -/
@[to_additive "An `AddSubgroup` of an `AddGroup` inherits an inverse."]
instance inv : Inv H :=
  ⟨fun a => ⟨a⁻¹, H.inv_mem a.2⟩⟩


/-- A subgroup of a group inherits a division -/
@[to_additive "An `AddSubgroup` of an `AddGroup` inherits a subtraction."]
instance div : Div H :=
  ⟨fun a b => ⟨a / b, H.div_mem a.2 b.2⟩⟩


/-- An `AddSubgroup` of an `AddGroup` inherits a natural scaling. -/
instance _root_.AddSubgroup.nsmul {G} [AddGroup G] {H : AddSubgroup G} : SMul ℕ H :=
  ⟨fun n a => ⟨n • a, H.nsmul_mem a.2 n⟩⟩


/-- A subgroup of a group inherits a natural power -/
@[to_additive existing]
protected instance npow : Pow H ℕ :=
  ⟨fun a n => ⟨a ^ n, H.pow_mem a.2 n⟩⟩


/-- An `AddSubgroup` of an `AddGroup` inherits an integer scaling. -/
instance _root_.AddSubgroup.zsmul {G} [AddGroup G] {H : AddSubgroup G} : SMul ℤ H :=
  ⟨fun n a => ⟨n • a, H.zsmul_mem a.2 n⟩⟩


/-- A subgroup of a group inherits an integer power -/
@[to_additive existing]
instance zpow : Pow H ℤ :=
  ⟨fun a n => ⟨a ^ n, H.zpow_mem a.2 n⟩⟩


@[to_additive (attr := simp, norm_cast)]
theorem coe_mul (x y : H) : (↑(x * y) : G) = ↑x * ↑y :=
  rfl


@[to_additive (attr := simp, norm_cast)]
theorem coe_one : ((1 : H) : G) = 1 :=
  rfl


@[to_additive (attr := simp, norm_cast)]
theorem coe_inv (x : H) : ↑(x⁻¹ : H) = (x⁻¹ : G) :=
  rfl


@[to_additive (attr := simp, norm_cast)]
theorem coe_div (x y : H) : (↑(x / y) : G) = ↑x / ↑y :=
  rfl

-- Porting note: removed simp, theorem has variable as head symbol

@[to_additive (attr := norm_cast)]
theorem coe_mk (x : G) (hx : x ∈ H) : ((⟨x, hx⟩ : H) : G) = x :=
  rfl


@[to_additive (attr := norm_cast)] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10685): dsimp can prove this
theorem coe_zpow (x : H) (n : ℤ) : ((x ^ n : H) : G) = (x : G) ^ n :=
  rfl


@[to_additive (attr := simp)]
theorem mk_eq_one {g : G} {h} : (⟨g, h⟩ : H) = 1 ↔ g = 1 := Submonoid.mk_eq_one ..


/-- A subgroup of a group inherits a group structure. -/
@[to_additive "An `AddSubgroup` of an `AddGroup` inherits an `AddGroup` structure."]
instance toGroup {G : Type*} [Group G] (H : Subgroup G) : Group H :=
  Subtype.coe_injective.group _ rfl (fun _ _ => rfl) (fun _ => rfl) (fun _ _ => rfl)
    (fun _ _ => rfl) fun _ _ => rfl


/-- A subgroup of a `CommGroup` is a `CommGroup`. -/
@[to_additive "An `AddSubgroup` of an `AddCommGroup` is an `AddCommGroup`."]
instance toCommGroup {G : Type*} [CommGroup G] (H : Subgroup G) : CommGroup H :=
  Subtype.coe_injective.commGroup _ rfl (fun _ _ => rfl) (fun _ => rfl) (fun _ _ => rfl)
    (fun _ _ => rfl) fun _ _ => rfl


/-- The natural group hom from a subgroup of group `G` to `G`. -/
@[to_additive "The natural group hom from an `AddSubgroup` of `AddGroup` `G` to `G`."]
protected def subtype : H →* G where
  toFun := ((↑) : H → G); map_one' := rfl; map_mul' _ _ := rfl


@[to_additive (attr := simp)]
theorem coeSubtype : ⇑ H.subtype = ((↑) : H → G) :=
  rfl


@[to_additive]
theorem subtype_injective : Function.Injective (Subgroup.subtype H) :=
  Subtype.coe_injective


/-- The inclusion homomorphism from a subgroup `H` contained in `K` to `K`. -/
@[to_additive "The inclusion homomorphism from an additive subgroup `H` contained in `K` to `K`."]
def inclusion {H K : Subgroup G} (h : H ≤ K) : H →* K :=
  MonoidHom.mk' (fun x => ⟨x, h x.2⟩) fun _ _ => rfl


@[to_additive (attr := simp)]
theorem coe_inclusion {H K : Subgroup G} {h : H ≤ K} (a : H) : (inclusion h a : G) = a := by
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    h : LE.le H K
    a : Subtype fun x => Membership.mem H x
    ⊢ Eq ↑((Subgroup.inclusion h) a) ↑a
  -/
  cases a
  /-
    case mk
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    h : LE.le H K
    val✝ : G
    property✝ : Membership.mem H val✝
    ⊢ Eq ↑((Subgroup.inclusion h) ⟨val✝, property✝⟩) ↑⟨val✝, property✝⟩
  -/
  simp only [inclusion, coe_mk, MonoidHom.mk'_apply]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem inclusion_injective {H K : Subgroup G} (h : H ≤ K) : Function.Injective <| inclusion h :=
  Set.inclusion_injective h


@[to_additive (attr := simp)]
lemma inclusion_inj {H K : Subgroup G} (h : H ≤ K) {x y : H} :
    inclusion h x = inclusion h y ↔ x = y :=
  (inclusion_injective h).eq_iff


@[to_additive (attr := simp)]
theorem subtype_comp_inclusion {H K : Subgroup G} (hH : H ≤ K) :
    K.subtype.comp (inclusion hH) = H.subtype :=
  rfl


/-- A subgroup is normal if whenever `n ∈ H`, then `g * n * g⁻¹ ∈ H` for every `g : G` -/
structure Normal : Prop where
  /-- `N` is closed under conjugation -/
  conj_mem : ∀ n, n ∈ H → ∀ g : G, g * n * g⁻¹ ∈ H


/-- An AddSubgroup is normal if whenever `n ∈ H`, then `g + n - g ∈ H` for every `g : G` -/
structure Normal (H : AddSubgroup A) : Prop where
  /-- `N` is closed under additive conjugation -/
  conj_mem : ∀ n, n ∈ H → ∀ g : A, g + n + -g ∈ H


@[to_additive]
instance (priority := 100) normal_of_comm {G : Type*} [CommGroup G] (H : Subgroup G) : H.Normal :=
      /-
        G✝ : Type u_1
        inst✝² : Group G✝
        A : Type u_2
        inst✝¹ : AddGroup A
        H✝ : Subgroup G✝
        G : Type u_3
        inst✝ : CommGroup G
        H : Subgroup G
        ⊢ ∀ (n : G), Membership.mem H n → ∀ (g : G), Membership.mem H (HMul.hMul (HMul …
      -/
  ⟨by simp [mul_comm, mul_left_comm]⟩
      /-
        🎉 no goals
      -/


@[to_additive]
theorem conj_mem' (nH : H.Normal) (n : G) (hn : n ∈ H) (g : G) :
    g⁻¹ * n * g ∈ H := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    nH : H.Normal
    n : G
    hn : Membership.mem H n
    g : G
    ⊢ Membership.mem H (HMul.hMul (HMul.hMul (Inv.inv g) n) g)
  -/
  convert nH.conj_mem n hn g⁻¹
  /-
    case h.e'_5.h.e'_6
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    nH : H.Normal
    n : G
    hn : Membership.mem H n
    g : G
    ⊢ Eq g (Inv.inv (Inv.inv g))
  -/
  rw [inv_inv]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mem_comm (nH : H.Normal) {a b : G} (h : a * b ∈ H) : b * a ∈ H := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    nH : H.Normal
    a b : G
    h : Membership.mem H (HMul.hMul a b)
    ⊢ Membership.mem H (HMul.hMul b a)
  -/
  have : a⁻¹ * (a * b) * a⁻¹⁻¹ ∈ H := nH.conj_mem (a * b) h a⁻¹
  -- Porting note: Previous code was:
  -- simpa
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    nH : H.Normal
    a b : G
    h : Membership.mem H (HMul.hMul a b)
    this : Membership.mem H (HMul.hMul (HMul.hMul (Inv.inv a) (HMul.hMul a b)) (In …
    ⊢ Membership.mem H (HMul.hMul b a)
  -/
  simp_all only [inv_mul_cancel_left, inv_inv]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mem_comm_iff (nH : H.Normal) {a b : G} : a * b ∈ H ↔ b * a ∈ H :=
  ⟨nH.mem_comm, nH.mem_comm⟩


/-- The `normalizer` of `H` is the largest subgroup of `G` inside which `H` is normal. -/
@[to_additive "The `normalizer` of `H` is the largest subgroup of `G` inside which `H` is normal."]
def normalizer : Subgroup G where
  carrier := { g : G | ∀ n, n ∈ H ↔ g * n * g⁻¹ ∈ H }
                 /-
                   G : Type u_1
                   inst✝¹ : Group G
                   A : Type u_2
                   inst✝ : AddGroup A
                   H : Subgroup G
                   ⊢ Membership.mem { carrier := setOf fun g => ∀ (n : G), Iff (Membership.mem H  …
                 -/
  one_mem' := by simp
    /-
      G : Type u_1
      inst✝¹ : Group G
      A : Type u_2
      inst✝ : AddGroup A
      H : Subgroup G
      a b : G
      ha : ∀ (n : G), Iff (Membership.mem H n) (Membership.mem H (HMul.hMul (HMul.hM …
      hb : ∀ (n : G), Iff (Membership.mem H n) (Membership.mem H (HMul.hMul (HMul.hM …
      n : G
      ⊢ Iff (Membership.mem H n) (Membership.mem H (HMul.hMul (HMul.hMul (HMul.hMul  …
    -/
                 /-
                   🎉 no goals
                 -/
    /-
      G : Type u_1
      inst✝¹ : Group G
      A : Type u_2
      inst✝ : AddGroup A
      H : Subgroup G
      a b : G
      ha : ∀ (n : G), Iff (Membership.mem H n) (Membership.mem H (HMul.hMul (HMul.hM …
      hb : ∀ (n : G), Iff (Membership.mem H n) (Membership.mem H (HMul.hMul (HMul.hM …
      n : G
      ⊢ Iff (Membership.mem H (HMul.hMul (HMul.hMul a (HMul.hMul (HMul.hMul b n) (In …
    -/
  mul_mem' {a b} (ha : ∀ n, n ∈ H ↔ a * n * a⁻¹ ∈ H) (hb : ∀ n, n ∈ H ↔ b * n * b⁻¹ ∈ H) n := by
    /-
      🎉 no goals
    -/
    rw [hb, ha]
    simp only [mul_assoc, mul_inv_rev]
  inv_mem' {a} (ha : ∀ n, n ∈ H ↔ a * n * a⁻¹ ∈ H) n := by
    /-
      G : Type u_1
      inst✝¹ : Group G
      A : Type u_2
      inst✝ : AddGroup A
      H : Subgroup G
      a : G
      ha : ∀ (n : G), Iff (Membership.mem H n) (Membership.mem H (HMul.hMul (HMul.hM …
      n : G
      ⊢ Iff (Membership.mem H n) (Membership.mem H (HMul.hMul (HMul.hMul (Inv.inv a) …
    -/
    rw [ha (a⁻¹ * n * a⁻¹⁻¹)]
    /-
      G : Type u_1
      inst✝¹ : Group G
      A : Type u_2
      inst✝ : AddGroup A
      H : Subgroup G
      a : G
      ha : ∀ (n : G), Iff (Membership.mem H n) (Membership.mem H (HMul.hMul (HMul.hM …
      n : G
      ⊢ Iff (Membership.mem H n) (Membership.mem H (HMul.hMul (HMul.hMul a (HMul.hMu …
    -/
    simp only [inv_inv, mul_assoc, mul_inv_cancel_left, mul_inv_cancel, mul_one]
    /-
      🎉 no goals
    -/

-- variant for sets.
-- TODO should this replace `normalizer`?

/-- The `setNormalizer` of `S` is the subgroup of `G` whose elements satisfy `g*S*g⁻¹=S` -/
@[to_additive
      "The `setNormalizer` of `S` is the subgroup of `G` whose elements satisfy
      `g+S-g=S`."]
def setNormalizer (S : Set G) : Subgroup G where
  carrier := { g : G | ∀ n, n ∈ S ↔ g * n * g⁻¹ ∈ S }
                 /-
                   G : Type u_1
                   inst✝¹ : Group G
                   A : Type u_2
                   inst✝ : AddGroup A
                   H : Subgroup G
                   S : Set G
                   ⊢ Membership.mem { carrier := setOf fun g => ∀ (n : G), Iff (Membership.mem S  …
                 -/
  one_mem' := by simp
    /-
      G : Type u_1
      inst✝¹ : Group G
      A : Type u_2
      inst✝ : AddGroup A
      H : Subgroup G
      S : Set G
      a b : G
      ha : ∀ (n : G), Iff (Membership.mem S n) (Membership.mem S (HMul.hMul (HMul.hM …
      hb : ∀ (n : G), Iff (Membership.mem S n) (Membership.mem S (HMul.hMul (HMul.hM …
      n : G
      ⊢ Iff (Membership.mem S n) (Membership.mem S (HMul.hMul (HMul.hMul (HMul.hMul  …
    -/
                 /-
                   🎉 no goals
                 -/
    /-
      G : Type u_1
      inst✝¹ : Group G
      A : Type u_2
      inst✝ : AddGroup A
      H : Subgroup G
      S : Set G
      a b : G
      ha : ∀ (n : G), Iff (Membership.mem S n) (Membership.mem S (HMul.hMul (HMul.hM …
      hb : ∀ (n : G), Iff (Membership.mem S n) (Membership.mem S (HMul.hMul (HMul.hM …
      n : G
      ⊢ Iff (Membership.mem S (HMul.hMul (HMul.hMul a (HMul.hMul (HMul.hMul b n) (In …
    -/
  mul_mem' {a b} (ha : ∀ n, n ∈ S ↔ a * n * a⁻¹ ∈ S) (hb : ∀ n, n ∈ S ↔ b * n * b⁻¹ ∈ S) n := by
    /-
      🎉 no goals
    -/
    rw [hb, ha]
    simp only [mul_assoc, mul_inv_rev]
  inv_mem' {a} (ha : ∀ n, n ∈ S ↔ a * n * a⁻¹ ∈ S) n := by
    /-
      G : Type u_1
      inst✝¹ : Group G
      A : Type u_2
      inst✝ : AddGroup A
      H : Subgroup G
      S : Set G
      a : G
      ha : ∀ (n : G), Iff (Membership.mem S n) (Membership.mem S (HMul.hMul (HMul.hM …
      n : G
      ⊢ Iff (Membership.mem S n) (Membership.mem S (HMul.hMul (HMul.hMul (Inv.inv a) …
    -/
    rw [ha (a⁻¹ * n * a⁻¹⁻¹)]
    /-
      G : Type u_1
      inst✝¹ : Group G
      A : Type u_2
      inst✝ : AddGroup A
      H : Subgroup G
      S : Set G
      a : G
      ha : ∀ (n : G), Iff (Membership.mem S n) (Membership.mem S (HMul.hMul (HMul.hM …
      n : G
      ⊢ Iff (Membership.mem S n) (Membership.mem S (HMul.hMul (HMul.hMul a (HMul.hMu …
    -/
    simp only [inv_inv, mul_assoc, mul_inv_cancel_left, mul_inv_cancel, mul_one]
    /-
      🎉 no goals
    -/


@[to_additive]
theorem mem_normalizer_iff {g : G} : g ∈ H.normalizer ↔ ∀ h, h ∈ H ↔ g * h * g⁻¹ ∈ H :=
  Iff.rfl


@[to_additive]
theorem mem_normalizer_iff'' {g : G} : g ∈ H.normalizer ↔ ∀ h : G, h ∈ H ↔ g⁻¹ * h * g ∈ H := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    g : G
    ⊢ Iff (Membership.mem H.normalizer g) (∀ (h : G), Iff (Membership.mem H h) (Me …
  -/
  rw [← inv_mem_iff (x := g), mem_normalizer_iff, inv_inv]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mem_normalizer_iff' {g : G} : g ∈ H.normalizer ↔ ∀ n, n * g ∈ H ↔ g * n ∈ H :=
                 /-
                   G : Type u_1
                   inst✝ : Group G
                   H : Subgroup G
                   g : G
                   h : Membership.mem H.normalizer g
                   n : G
                   ⊢ Iff (Membership.mem H (HMul.hMul n g)) (Membership.mem H (HMul.hMul g n))
                 -/
  ⟨fun h n => by rw [h, mul_assoc, mul_inv_cancel_right], fun h n => by
                 /-
                   🎉 no goals
                 -/
    /-
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      g : G
      h : ∀ (n : G), Iff (Membership.mem H (HMul.hMul n g)) (Membership.mem H (HMul. …
      n : G
      ⊢ Iff (Membership.mem H n) (Membership.mem H (HMul.hMul (HMul.hMul g n) (Inv.i …
    -/
    rw [mul_assoc, ← h, inv_mul_cancel_right]⟩
    /-
      🎉 no goals
    -/


@[to_additive]
theorem le_normalizer : H ≤ normalizer H := fun x xH n => by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    x : G
    xH : Membership.mem H x
    n : G
    ⊢ Iff (Membership.mem H n) (Membership.mem H (HMul.hMul (HMul.hMul x n) (Inv.i …
  -/
  rw [H.mul_mem_cancel_right (H.inv_mem xH), H.mul_mem_cancel_left xH]
  /-
    🎉 no goals
  -/


/-- Commutativity of a subgroup -/
structure IsCommutative : Prop where
  /-- `*` is commutative on `H` -/
  is_comm : Std.Commutative (α := H) (· * ·)


/-- Commutativity of an additive subgroup -/
structure _root_.AddSubgroup.IsCommutative (H : AddSubgroup A) : Prop where
  /-- `+` is commutative on `H` -/
  is_comm : Std.Commutative (α := H) (· + ·)


/-- A commutative subgroup is commutative. -/
@[to_additive "A commutative subgroup is commutative."]
instance IsCommutative.commGroup [h : H.IsCommutative] : CommGroup H :=
  { H.toGroup with mul_comm := h.is_comm.comm }


/-- A subgroup of a commutative group is commutative. -/
@[to_additive "A subgroup of a commutative group is commutative."]
instance commGroup_isCommutative {G : Type*} [CommGroup G] (H : Subgroup G) : H.IsCommutative :=
  ⟨CommMagma.to_isCommutative⟩


@[to_additive]
lemma mul_comm_of_mem_isCommutative [H.IsCommutative] {a b : G} (ha : a ∈ H) (hb : b ∈ H) :
    a * b = b * a := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    inst✝ : H.IsCommutative
    a b : G
    ha : Membership.mem H a
    hb : Membership.mem H b
    ⊢ Eq (HMul.hMul a b) (HMul.hMul b a)
  -/
  simpa only [MulMemClass.mk_mul_mk, Subtype.mk.injEq] using mul_comm (⟨a, ha⟩ : H) (⟨b, hb⟩ : H)
  /-
    🎉 no goals
  -/


