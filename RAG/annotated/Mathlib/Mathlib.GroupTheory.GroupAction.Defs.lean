/-- The orbit of an element under an action. -/
@[to_additive "The orbit of an element under an action."]
def orbit (a : α) :=
  Set.range fun m : M => m • a


@[to_additive]
theorem mem_orbit_iff {a₁ a₂ : α} : a₂ ∈ orbit M a₁ ↔ ∃ x : M, x • a₁ = a₂ :=
  Iff.rfl


@[to_additive (attr := simp)]
theorem mem_orbit (a : α) (m : M) : m • a ∈ orbit M a :=
  ⟨m, rfl⟩


@[to_additive (attr := simp)]
theorem mem_orbit_self (a : α) : a ∈ orbit M a :=
         /-
           M : Type u
           inst✝¹ : Monoid M
           α : Type v
           inst✝ : MulAction M α
           a : α
           ⊢ Eq ((fun m => HSMul.hSMul m a) 1) a
         -/
  ⟨1, by simp [MulAction.one_smul]⟩
         /-
           🎉 no goals
         -/


@[to_additive]
theorem orbit_nonempty (a : α) : Set.Nonempty (orbit M a) :=
  Set.range_nonempty _


@[to_additive]
theorem mapsTo_smul_orbit (m : M) (a : α) : Set.MapsTo (m • ·) (orbit M a) (orbit M a) :=
  Set.range_subset_iff.2 fun m' => ⟨m * m', mul_smul _ _ _⟩


@[to_additive]
theorem smul_orbit_subset (m : M) (a : α) : m • orbit M a ⊆ orbit M a :=
  (mapsTo_smul_orbit m a).image_subset


@[to_additive]
theorem orbit_smul_subset (m : M) (a : α) : orbit M (m • a) ⊆ orbit M a :=
  Set.range_subset_iff.2 fun m' => mul_smul m' m a ▸ mem_orbit _ _


@[to_additive]
instance {a : α} : MulAction M (orbit M a) where
  smul m := (mapsTo_smul_orbit m a).restrict _ _ _
  one_smul m := Subtype.ext (one_smul M (m : α))
  mul_smul m m' a' := Subtype.ext (mul_smul m m' (a' : α))


@[to_additive (attr := simp)]
theorem orbit.coe_smul {a : α} {m : M} {a' : orbit M a} : ↑(m • a') = m • (a' : α) :=
  rfl


@[to_additive]
lemma orbit_submonoid_subset (S : Submonoid M) (a : α) : orbit S a ⊆ orbit M a := by
  /-
    M : Type u
    inst✝¹ : Monoid M
    α : Type v
    inst✝ : MulAction M α
    S : Submonoid M
    a : α
    ⊢ HasSubset.Subset (MulAction.orbit (Subtype fun x => Membership.mem S x) a) ( …
  -/
  rintro b ⟨g, rfl⟩
  /-
    case intro
    M : Type u
    inst✝¹ : Monoid M
    α : Type v
    inst✝ : MulAction M α
    S : Submonoid M
    a : α
    g : Subtype fun x => Membership.mem S x
    ⊢ Membership.mem (MulAction.orbit M a) ((fun m => HSMul.hSMul m a) g)
  -/
  exact mem_orbit _ _
  /-
    🎉 no goals
  -/


@[to_additive]
lemma mem_orbit_of_mem_orbit_submonoid {S : Submonoid M} {a b : α} (h : a ∈ orbit S b) :
    a ∈ orbit M b :=
  orbit_submonoid_subset S _ h


/-- The set of elements fixed under the whole action. -/
@[to_additive "The set of elements fixed under the whole action."]
def fixedPoints : Set α :=
  { a : α | ∀ m : M, m • a = a }


/-- `fixedBy m` is the set of elements fixed by `m`. -/
@[to_additive "`fixedBy m` is the set of elements fixed by `m`."]
def fixedBy (m : M) : Set α :=
  { x | m • x = x }


@[to_additive]
theorem fixed_eq_iInter_fixedBy : fixedPoints M α = ⋂ m : M, fixedBy α m :=
  Set.ext fun _ =>
    ⟨fun hx => Set.mem_iInter.2 fun m => hx m, fun hx m => (Set.mem_iInter.1 hx m : _)⟩


@[to_additive (attr := simp)]
theorem mem_fixedPoints {a : α} : a ∈ fixedPoints M α ↔ ∀ m : M, m • a = a :=
  Iff.rfl


@[to_additive (attr := simp)]
theorem mem_fixedBy {m : M} {a : α} : a ∈ fixedBy α m ↔ m • a = a :=
  Iff.rfl


@[to_additive]
theorem mem_fixedPoints' {a : α} : a ∈ fixedPoints M α ↔ ∀ a', a' ∈ orbit M a → a' = a :=
  ⟨fun h _ h₁ =>
    let ⟨m, hm⟩ := mem_orbit_iff.1 h₁
    hm ▸ h m,
    fun h _ => h _ (mem_orbit _ _)⟩


/-- The stabilizer of a point `a` as a submonoid of `M`. -/
@[to_additive "The stabilizer of a point `a` as an additive submonoid of `M`."]
def stabilizerSubmonoid (a : α) : Submonoid M where
  carrier := { m | m • a = a }
  one_mem' := one_smul _ a
  mul_mem' {m m'} (ha : m • a = a) (hb : m' • a = a) :=
                             /-
                               M : Type u
                               inst✝² : Monoid M
                               α : Type v
                               inst✝¹ : MulAction M α
                               β : Type u_1
                               inst✝ : MulAction M β
                               a : α
                               m m' : M
                               ha : Eq (HSMul.hSMul m a) a
                               hb : Eq (HSMul.hSMul m' a) a
                               ⊢ Eq (HSMul.hSMul (HMul.hMul m m') a) a
                             -/
    show (m * m') • a = a by rw [← smul_smul, hb, ha]
                             /-
                               🎉 no goals
                             -/


@[to_additive]
instance [DecidableEq α] (a : α) : DecidablePred (· ∈ stabilizerSubmonoid M a) :=
  fun _ => inferInstanceAs <| Decidable (_ = _)


@[to_additive (attr := simp)]
theorem mem_stabilizerSubmonoid_iff {a : α} {m : M} : m ∈ stabilizerSubmonoid M a ↔ m • a = a :=
  Iff.rfl


/-- The submonoid of elements fixed under the whole action. -/
def FixedPoints.submonoid : Submonoid α where
  carrier := MulAction.fixedPoints M α
  one_mem' := smul_one
                         /-
                           M : Type u
                           α : Type v
                           inst✝² : Monoid M
                           inst✝¹ : Monoid α
                           inst✝ : MulDistribMulAction M α
                           a✝ b✝ : α
                           ha : Membership.mem (MulAction.fixedPoints M α) a✝
                           hb : Membership.mem (MulAction.fixedPoints M α) b✝
                           x✝ : M
                           ⊢ Eq (HSMul.hSMul x✝ (HMul.hMul a✝ b✝)) (HMul.hMul a✝ b✝)
                         -/
  mul_mem' ha hb _ := by rw [smul_mul', ha, hb]
                         /-
                           🎉 no goals
                         -/


@[simp]
lemma FixedPoints.mem_submonoid (a : α) : a ∈ submonoid M α ↔ ∀ m : M, m • a = a :=
  Iff.rfl


/-- The subgroup of elements fixed under the whole action. -/
def subgroup : Subgroup α where
  __ := submonoid M α
                      /-
                        M : Type u
                        α : Type v
                        inst✝² : Monoid M
                        inst✝¹ : Group α
                        inst✝ : MulDistribMulAction M α
                        x✝¹ : α
                        ha : Membership.mem __spread✝⁻⁰.carrier x✝¹
                        x✝ : M
                        ⊢ Eq (HSMul.hSMul x✝ (Inv.inv x✝¹)) (Inv.inv x✝¹)
                      -/
  inv_mem' ha _ := by rw [smul_inv', ha]
                      /-
                        🎉 no goals
                      -/


/-- The notation for `FixedPoints.subgroup`, chosen to resemble `αᴹ`. -/
scoped notation α "^*" M:51 => FixedPoints.subgroup M α


@[simp]
lemma mem_subgroup (a : α) : a ∈ α^*M ↔ ∀ m : M, m • a = a :=
  Iff.rfl


@[simp]
lemma subgroup_toSubmonoid : (α^*M).toSubmonoid = submonoid M α :=
  rfl


/-- The additive submonoid of elements fixed under the whole action. -/
def FixedPoints.addSubmonoid : AddSubmonoid α where
  carrier := MulAction.fixedPoints M α
  zero_mem' := smul_zero
                         /-
                           M : Type u
                           α : Type v
                           inst✝² : Monoid M
                           inst✝¹ : AddMonoid α
                           inst✝ : DistribMulAction M α
                           a✝ b✝ : α
                           ha : Membership.mem (MulAction.fixedPoints M α) a✝
                           hb : Membership.mem (MulAction.fixedPoints M α) b✝
                           x✝ : M
                           ⊢ Eq (HSMul.hSMul x✝ (HAdd.hAdd a✝ b✝)) (HAdd.hAdd a✝ b✝)
                         -/
  add_mem' ha hb _ := by rw [smul_add, ha, hb]
                         /-
                           🎉 no goals
                         -/


@[simp]
lemma FixedPoints.mem_addSubmonoid (a : α) : a ∈ addSubmonoid M α ↔ ∀ m : M, m • a = a :=
  Iff.rfl


/-- The additive subgroup of elements fixed under the whole action. -/
def FixedPoints.addSubgroup : AddSubgroup α where
  __ := addSubmonoid M α
                      /-
                        M : Type u
                        α : Type v
                        inst✝² : Monoid M
                        inst✝¹ : AddGroup α
                        inst✝ : DistribMulAction M α
                        x✝¹ : α
                        ha : Membership.mem __spread✝⁻⁰.carrier x✝¹
                        x✝ : M
                        ⊢ Eq (HSMul.hSMul x✝ (Neg.neg x✝¹)) (Neg.neg x✝¹)
                      -/
  neg_mem' ha _ := by rw [smul_neg, ha]
                      /-
                        🎉 no goals
                      -/


/-- The notation for `FixedPoints.addSubgroup`, chosen to resemble `αᴹ`. -/
notation α "^+" M:51 => FixedPoints.addSubgroup M α


@[simp]
lemma FixedPoints.mem_addSubgroup (a : α) : a ∈ α^+M ↔ ∀ m : M, m • a = a :=
  Iff.rfl


@[simp]
lemma FixedPoints.addSubgroup_toAddSubmonoid : (α^+M).toAddSubmonoid = addSubmonoid M α :=
  rfl


@[to_additive (attr := simp)]
theorem orbit_smul (g : G) (a : α) : orbit G (g • a) = orbit G a :=
  (orbit_smul_subset g a).antisymm <|
    calc
                                              /-
                                                G : Type u_1
                                                α : Type u_2
                                                inst✝¹ : Group G
                                                inst✝ : MulAction G α
                                                g : G
                                                a : α
                                                ⊢ Eq (MulAction.orbit G a) (MulAction.orbit G (HSMul.hSMul (Inv.inv g) (HSMul. …
                                              -/
      orbit G a = orbit G (g⁻¹ • g • a) := by rw [inv_smul_smul]
                                              /-
                                                🎉 no goals
                                              -/
      _ ⊆ orbit G (g • a) := orbit_smul_subset _ _


@[to_additive]
theorem orbit_eq_iff {a b : α} : orbit G a = orbit G b ↔ a ∈ orbit G b :=
  ⟨fun h => h ▸ mem_orbit_self _, fun ⟨_, hc⟩ => hc ▸ orbit_smul _ _⟩


@[to_additive]
theorem mem_orbit_smul (g : G) (a : α) : a ∈ orbit G (g • a) := by
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    g : G
    a : α
    ⊢ Membership.mem (MulAction.orbit G (HSMul.hSMul g a)) a
  -/
  simp only [orbit_smul, mem_orbit_self]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem smul_mem_orbit_smul (g h : G) (a : α) : g • a ∈ orbit G (h • a) := by
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    g h : G
    a : α
    ⊢ Membership.mem (MulAction.orbit G (HSMul.hSMul h a)) (HSMul.hSMul g a)
  -/
  simp only [orbit_smul, mem_orbit]
  /-
    🎉 no goals
  -/


@[to_additive]
instance instMulAction (H : Subgroup G) : MulAction H α :=
  inferInstanceAs (MulAction H.toSubmonoid α)


@[to_additive]
lemma orbit_subgroup_subset (H : Subgroup G) (a : α) : orbit H a ⊆ orbit G a :=
  orbit_submonoid_subset H.toSubmonoid a


@[to_additive]
lemma mem_orbit_of_mem_orbit_subgroup {H : Subgroup G} {a b : α} (h : a ∈ orbit H b) :
    a ∈ orbit G b :=
  orbit_subgroup_subset H _ h


@[to_additive]
lemma mem_orbit_symm {a₁ a₂ : α} : a₁ ∈ orbit G a₂ ↔ a₂ ∈ orbit G a₁ := by
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    a₁ a₂ : α
    ⊢ Iff (Membership.mem (MulAction.orbit G a₂) a₁) (Membership.mem (MulAction.or …
  -/
  simp_rw [← orbit_eq_iff, eq_comm]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma mem_subgroup_orbit_iff {H : Subgroup G} {x : α} {a b : orbit G x} :
    a ∈ MulAction.orbit H b ↔ (a : α) ∈ MulAction.orbit H (b : α) := by
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    H : Subgroup G
    x : α
    a b : ↑(MulAction.orbit G x)
    ⊢ Iff (Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem H x) b …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ?_⟩
    /-
      case refine_1
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      H : Subgroup G
      x : α
      a b : ↑(MulAction.orbit G x)
      h : Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem H x) b) a
      ⊢ Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem H x) ↑b) ↑a
    -/
  · rcases h with ⟨g, rfl⟩
    /-
      case refine_1.intro
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      H : Subgroup G
      x : α
      b : ↑(MulAction.orbit G x)
      g : Subtype fun x => Membership.mem H x
      ⊢ Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem H x) ↑b) ↑( …
    -/
    exact MulAction.mem_orbit _ g
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      H : Subgroup G
      x : α
      a b : ↑(MulAction.orbit G x)
      h : Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem H x) ↑b) ↑a
      ⊢ Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem H x) b) a
    -/
  · rcases h with ⟨g, h⟩
    /-
      case refine_2.intro
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      H : Subgroup G
      x : α
      a b : ↑(MulAction.orbit G x)
      g : Subtype fun x => Membership.mem H x
      h : Eq ((fun m => HSMul.hSMul m ↑b) g) ↑a
      ⊢ Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem H x) b) a
    -/
    dsimp at h
    /-
      case refine_2.intro
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      H : Subgroup G
      x : α
      a b : ↑(MulAction.orbit G x)
      g : Subtype fun x => Membership.mem H x
      h : Eq (HSMul.hSMul g ↑b) ↑a
      ⊢ Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem H x) b) a
    -/
    erw [← orbit.coe_smul, ← Subtype.ext_iff] at h
    /-
      case refine_2.intro
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      H : Subgroup G
      x : α
      a b : ↑(MulAction.orbit G x)
      g : Subtype fun x => Membership.mem H x
      h : Eq (HSMul.hSMul (H.subtype g) b) a
      ⊢ Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem H x) b) a
    -/
    subst h
    /-
      case refine_2.intro
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      H : Subgroup G
      x : α
      b : ↑(MulAction.orbit G x)
      g : Subtype fun x => Membership.mem H x
      ⊢ Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem H x) b) (HS …
    -/
    exact MulAction.mem_orbit _ g
    /-
      🎉 no goals
    -/


/-- The relation 'in the same orbit'. -/
@[to_additive "The relation 'in the same orbit'."]
def orbitRel : Setoid α where
  r a b := a ∈ orbit G b
  iseqv :=
                                     /-
                                       G : Type u_1
                                       α : Type u_2
                                       β : Type u_3
                                       inst✝² : Group G
                                       inst✝¹ : MulAction G α
                                       inst✝ : MulAction G β
                                       a b : α
                                       ⊢ Membership.mem (MulAction.orbit G b) a → Membership.mem (MulAction.orbit G a …
                                     -/
    ⟨mem_orbit_self, fun {a b} => by simp [orbit_eq_iff.symm, eq_comm], fun {a b} => by
                                     /-
                                       🎉 no goals
                                     -/
      /-
        G : Type u_1
        α : Type u_2
        β : Type u_3
        inst✝² : Group G
        inst✝¹ : MulAction G α
        inst✝ : MulAction G β
        a b : α
        ⊢ ∀ {z : α}, Membership.mem (MulAction.orbit G b) a → Membership.mem (MulActio …
      -/
      simp +contextual [orbit_eq_iff.symm, eq_comm]⟩
      /-
        🎉 no goals
      -/


@[to_additive]
theorem orbitRel_apply {a b : α} : orbitRel G α a b ↔ a ∈ orbit G b :=
  Iff.rfl


@[to_additive]
alias orbitRel_r_apply := orbitRel_apply

-- `alias` doesn't add the deprecation suggestion to the `to_additive` version
-- see https://github.com/leanprover-community/mathlib4/issues/19424

/-- When you take a set `U` in `α`, push it down to the quotient, and pull back, you get the union
of the orbit of `U` under `G`. -/
@[to_additive
      "When you take a set `U` in `α`, push it down to the quotient, and pull back, you get the
      union of the orbit of `U` under `G`."]
theorem quotient_preimage_image_eq_union_mul (U : Set α) :
    letI := orbitRel G α
    Quotient.mk' ⁻¹' (Quotient.mk' '' U) = ⋃ g : G, (g • ·) '' U := by
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    U : Set α
    ⊢ Eq (Set.preimage Quotient.mk' (Set.image Quotient.mk' U)) (Set.iUnion fun g  …
  -/
  letI := orbitRel G α
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    U : Set α
    this : Setoid α := MulAction.orbitRel G α
    ⊢ Eq (Set.preimage Quotient.mk' (Set.image Quotient.mk' U)) (Set.iUnion fun g  …
  -/
  set f : α → Quotient (MulAction.orbitRel G α) := Quotient.mk'
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    U : Set α
    this : Setoid α := MulAction.orbitRel G α
    f : α → Quotient (MulAction.orbitRel G α) := Quotient.mk'
    ⊢ Eq (Set.preimage f (Set.image f U)) (Set.iUnion fun g => Set.image (fun x => …
  -/
  ext a
  /-
    case h
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    U : Set α
    this : Setoid α := MulAction.orbitRel G α
    f : α → Quotient (MulAction.orbitRel G α) := Quotient.mk'
    a : α
    ⊢ Iff (Membership.mem (Set.preimage f (Set.image f U)) a) (Membership.mem (Set …
  -/
  constructor
    /-
      case h.mp
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      U : Set α
      this : Setoid α := MulAction.orbitRel G α
      f : α → Quotient (MulAction.orbitRel G α) := Quotient.mk'
      a : α
      ⊢ Membership.mem (Set.preimage f (Set.image f U)) a → Membership.mem (Set.iUni …
    -/
  · rintro ⟨b, hb, hab⟩
    /-
      case h.mp.intro.intro
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      U : Set α
      this : Setoid α := MulAction.orbitRel G α
      f : α → Quotient (MulAction.orbitRel G α) := Quotient.mk'
      a b : α
      hb : Membership.mem U b
      hab : Eq (f b) (f a)
      ⊢ Membership.mem (Set.iUnion fun g => Set.image (fun x => HSMul.hSMul g x) U) a
    -/
    obtain ⟨g, rfl⟩ := Quotient.exact hab
    /-
      case h.mp.intro.intro.intro
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      U : Set α
      this : Setoid α := MulAction.orbitRel G α
      f : α → Quotient (MulAction.orbitRel G α) := Quotient.mk'
      a : α
      g : G
      hb : Membership.mem U ((fun m => HSMul.hSMul m a) g)
      hab : Eq (f ((fun m => HSMul.hSMul m a) g)) (f a)
      ⊢ Membership.mem (Set.iUnion fun g => Set.image (fun x => HSMul.hSMul g x) U) a
    -/
    rw [Set.mem_iUnion]
    /-
      case h.mp.intro.intro.intro
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      U : Set α
      this : Setoid α := MulAction.orbitRel G α
      f : α → Quotient (MulAction.orbitRel G α) := Quotient.mk'
      a : α
      g : G
      hb : Membership.mem U ((fun m => HSMul.hSMul m a) g)
      hab : Eq (f ((fun m => HSMul.hSMul m a) g)) (f a)
      ⊢ Exists fun i => Membership.mem (Set.image (fun x => HSMul.hSMul i x) U) a
    -/
    exact ⟨g⁻¹, g • a, hb, inv_smul_smul g a⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      U : Set α
      this : Setoid α := MulAction.orbitRel G α
      f : α → Quotient (MulAction.orbitRel G α) := Quotient.mk'
      a : α
      ⊢ Membership.mem (Set.iUnion fun g => Set.image (fun x => HSMul.hSMul g x) U)  …
    -/
  · intro hx
    /-
      case h.mpr
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      U : Set α
      this : Setoid α := MulAction.orbitRel G α
      f : α → Quotient (MulAction.orbitRel G α) := Quotient.mk'
      a : α
      hx : Membership.mem (Set.iUnion fun g => Set.image (fun x => HSMul.hSMul g x)  …
      ⊢ Membership.mem (Set.preimage f (Set.image f U)) a
    -/
    rw [Set.mem_iUnion] at hx
    /-
      case h.mpr
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      U : Set α
      this : Setoid α := MulAction.orbitRel G α
      f : α → Quotient (MulAction.orbitRel G α) := Quotient.mk'
      a : α
      hx : Exists fun i => Membership.mem (Set.image (fun x => HSMul.hSMul i x) U) a
      ⊢ Membership.mem (Set.preimage f (Set.image f U)) a
    -/
    obtain ⟨g, u, hu₁, hu₂⟩ := hx
    /-
      case h.mpr.intro.intro.intro
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      U : Set α
      this : Setoid α := MulAction.orbitRel G α
      f : α → Quotient (MulAction.orbitRel G α) := Quotient.mk'
      a : α
      g : G
      u : α
      hu₁ : Membership.mem U u
      hu₂ : Eq ((fun x => HSMul.hSMul g x) u) a
      ⊢ Membership.mem (Set.preimage f (Set.image f U)) a
    -/
    rw [Set.mem_preimage, Set.mem_image]
    /-
      case h.mpr.intro.intro.intro
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      U : Set α
      this : Setoid α := MulAction.orbitRel G α
      f : α → Quotient (MulAction.orbitRel G α) := Quotient.mk'
      a : α
      g : G
      u : α
      hu₁ : Membership.mem U u
      hu₂ : Eq ((fun x => HSMul.hSMul g x) u) a
      ⊢ Exists fun x => And (Membership.mem U x) (Eq (f x) (f a))
    -/
    refine ⟨g⁻¹ • a, ?_, by simp only [f, Quotient.eq']; use g⁻¹⟩
    /-
      case h.mpr.intro.intro.intro
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      U : Set α
      this : Setoid α := MulAction.orbitRel G α
      f : α → Quotient (MulAction.orbitRel G α) := Quotient.mk'
      a : α
      g : G
      u : α
      hu₁ : Membership.mem U u
      hu₂ : Eq ((fun x => HSMul.hSMul g x) u) a
      ⊢ Membership.mem U (HSMul.hSMul (Inv.inv g) a)
    -/
    rw [← hu₂]
    /-
      case h.mpr.intro.intro.intro
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      U : Set α
      this : Setoid α := MulAction.orbitRel G α
      f : α → Quotient (MulAction.orbitRel G α) := Quotient.mk'
      a : α
      g : G
      u : α
      hu₁ : Membership.mem U u
      hu₂ : Eq ((fun x => HSMul.hSMul g x) u) a
      ⊢ Membership.mem U (HSMul.hSMul (Inv.inv g) ((fun x => HSMul.hSMul g x) u))
    -/
    convert hu₁
    /-
      case h.e'_5
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      U : Set α
      this : Setoid α := MulAction.orbitRel G α
      f : α → Quotient (MulAction.orbitRel G α) := Quotient.mk'
      a : α
      g : G
      u : α
      hu₁ : Membership.mem U u
      hu₂ : Eq ((fun x => HSMul.hSMul g x) u) a
      ⊢ Eq (HSMul.hSMul (Inv.inv g) ((fun x => HSMul.hSMul g x) u)) u
    -/
    simp only [inv_smul_smul]
    /-
      🎉 no goals
    -/


@[to_additive]
theorem disjoint_image_image_iff {U V : Set α} :
    letI := orbitRel G α
    Disjoint (Quotient.mk' '' U) (Quotient.mk' '' V) ↔ ∀ x ∈ U, ∀ g : G, g • x ∉ V := by
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    U V : Set α
    ⊢ Iff (Disjoint (Set.image Quotient.mk' U) (Set.image Quotient.mk' V)) (∀ (x : …
  -/
  letI := orbitRel G α
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    U V : Set α
    this : Setoid α := MulAction.orbitRel G α
    ⊢ Iff (Disjoint (Set.image Quotient.mk' U) (Set.image Quotient.mk' V)) (∀ (x : …
  -/
  set f : α → Quotient (MulAction.orbitRel G α) := Quotient.mk'
  refine
    ⟨fun h a a_in_U g g_in_V =>
      h.le_bot ⟨⟨a, a_in_U, Quotient.sound ⟨g⁻¹, ?_⟩⟩, ⟨g • a, g_in_V, rfl⟩⟩, ?_⟩
    /-
      case refine_1
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      U V : Set α
      this : Setoid α := MulAction.orbitRel G α
      f : α → Quotient (MulAction.orbitRel G α) := Quotient.mk'
      h : Disjoint (Set.image f U) (Set.image f V)
      a : α
      a_in_U : Membership.mem U a
      g : G
      g_in_V : Membership.mem V (HSMul.hSMul g a)
      ⊢ Eq ((fun m => HSMul.hSMul m (HSMul.hSMul g a)) (Inv.inv g)) a
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      U V : Set α
      this : Setoid α := MulAction.orbitRel G α
      f : α → Quotient (MulAction.orbitRel G α) := Quotient.mk'
      ⊢ (∀ (x : α), Membership.mem U x → ∀ (g : G), Not (Membership.mem V (HSMul.hSM …
    -/
  · intro h
    /-
      case refine_2
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      U V : Set α
      this : Setoid α := MulAction.orbitRel G α
      f : α → Quotient (MulAction.orbitRel G α) := Quotient.mk'
      h : ∀ (x : α), Membership.mem U x → ∀ (g : G), Not (Membership.mem V (HSMul.hS …
      ⊢ Disjoint (Set.image f U) (Set.image f V)
    -/
    rw [Set.disjoint_left]
    /-
      case refine_2
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      U V : Set α
      this : Setoid α := MulAction.orbitRel G α
      f : α → Quotient (MulAction.orbitRel G α) := Quotient.mk'
      h : ∀ (x : α), Membership.mem U x → ∀ (g : G), Not (Membership.mem V (HSMul.hS …
      ⊢ ∀ ⦃a : Quotient (MulAction.orbitRel G α)⦄, Membership.mem (Set.image f U) a  …
    -/
    rintro _ ⟨b, hb₁, hb₂⟩ ⟨c, hc₁, hc₂⟩
    /-
      case refine_2.intro.intro.intro.intro
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      U V : Set α
      this : Setoid α := MulAction.orbitRel G α
      f : α → Quotient (MulAction.orbitRel G α) := Quotient.mk'
      h : ∀ (x : α), Membership.mem U x → ∀ (g : G), Not (Membership.mem V (HSMul.hS …
      a✝ : Quotient (MulAction.orbitRel G α)
      b : α
      hb₁ : Membership.mem U b
      hb₂ : Eq (f b) a✝
      c : α
      hc₁ : Membership.mem V c
      hc₂ : Eq (f c) a✝
      ⊢ False
    -/
    obtain ⟨g, rfl⟩ := Quotient.exact (hc₂.trans hb₂.symm)
    /-
      case refine_2.intro.intro.intro.intro.intro
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      U V : Set α
      this : Setoid α := MulAction.orbitRel G α
      f : α → Quotient (MulAction.orbitRel G α) := Quotient.mk'
      h : ∀ (x : α), Membership.mem U x → ∀ (g : G), Not (Membership.mem V (HSMul.hS …
      a✝ : Quotient (MulAction.orbitRel G α)
      b : α
      hb₁ : Membership.mem U b
      hb₂ : Eq (f b) a✝
      g : G
      hc₁ : Membership.mem V ((fun m => HSMul.hSMul m b) g)
      hc₂ : Eq (f ((fun m => HSMul.hSMul m b) g)) a✝
      ⊢ False
    -/
    exact h b hb₁ g hc₁
    /-
      🎉 no goals
    -/


@[to_additive]
theorem image_inter_image_iff (U V : Set α) :
    letI := orbitRel G α
    Quotient.mk' '' U ∩ Quotient.mk' '' V = ∅ ↔ ∀ x ∈ U, ∀ g : G, g • x ∉ V :=
  Set.disjoint_iff_inter_eq_empty.symm.trans disjoint_image_image_iff


/-- The quotient by `MulAction.orbitRel`, given a name to enable dot notation. -/
@[to_additive
    "The quotient by `AddAction.orbitRel`, given a name to enable dot notation."]
abbrev orbitRel.Quotient : Type _ :=
  _root_.Quotient <| orbitRel G α


/-- The orbit corresponding to an element of the quotient by `MulAction.orbitRel` -/
@[to_additive "The orbit corresponding to an element of the quotient by `AddAction.orbitRel`"]
nonrec def orbitRel.Quotient.orbit (x : orbitRel.Quotient G α) : Set α :=
  Quotient.liftOn' x (orbit G) fun _ _ => MulAction.orbit_eq_iff.2


@[to_additive (attr := simp)]
theorem orbitRel.Quotient.orbit_mk (a : α) :
    orbitRel.Quotient.orbit (Quotient.mk'' a : orbitRel.Quotient G α) = MulAction.orbit G a :=
  rfl


@[to_additive]
theorem orbitRel.Quotient.mem_orbit {a : α} {x : orbitRel.Quotient G α} :
    a ∈ x.orbit ↔ Quotient.mk'' a = x := by
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    a : α
    x : MulAction.orbitRel.Quotient G α
    ⊢ Iff (Membership.mem x.orbit a) (Eq (Quotient.mk'' a) x)
  -/
  induction x using Quotient.inductionOn'
  /-
    case h
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    a a✝ : α
    ⊢ Iff (Membership.mem (MulAction.orbitRel.Quotient.orbit (Quotient.mk'' a✝)) a …
  -/
  rw [Quotient.eq'']
  /-
    case h
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    a a✝ : α
    ⊢ Iff (Membership.mem (MulAction.orbitRel.Quotient.orbit (Quotient.mk'' a✝)) a …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Note that `hφ = Quotient.out_eq'` is a useful choice here. -/
@[to_additive "Note that `hφ = Quotient.out_eq'` is a useful choice here."]
theorem orbitRel.Quotient.orbit_eq_orbit_out (x : orbitRel.Quotient G α)
    {φ : orbitRel.Quotient G α → α} (hφ : letI := orbitRel G α; RightInverse φ Quotient.mk') :
    orbitRel.Quotient.orbit x = MulAction.orbit G (φ x) := by
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    x : MulAction.orbitRel.Quotient G α
    φ : MulAction.orbitRel.Quotient G α → α
    hφ : Function.RightInverse φ Quotient.mk'
    ⊢ Eq x.orbit (MulAction.orbit G (φ x))
  -/
  conv_lhs => rw [← hφ x]
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    x : MulAction.orbitRel.Quotient G α
    φ : MulAction.orbitRel.Quotient G α → α
    hφ : Function.RightInverse φ Quotient.mk'
    ⊢ Eq (MulAction.orbitRel.Quotient.orbit (Quotient.mk' (φ x))) (MulAction.orbit …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[to_additive]
lemma orbitRel.Quotient.orbit_injective :
    Injective (orbitRel.Quotient.orbit : orbitRel.Quotient G α → Set α) := by
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    ⊢ Function.Injective MulAction.orbitRel.Quotient.orbit
  -/
  intro x y h
  simp_rw [orbitRel.Quotient.orbit_eq_orbit_out _ Quotient.out_eq', orbit_eq_iff,
    ← orbitRel_apply] at h
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    x y : MulAction.orbitRel.Quotient G α
    h : (MulAction.orbitRel G α) (Quotient.out x) (Quotient.out y)
    ⊢ Eq x y
  -/
  simpa [← Quotient.eq''] using h
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma orbitRel.Quotient.orbit_inj {x y : orbitRel.Quotient G α} : x.orbit = y.orbit ↔ x = y :=
  orbitRel.Quotient.orbit_injective.eq_iff


@[to_additive]
lemma orbitRel.quotient_eq_of_quotient_subgroup_eq {H : Subgroup G} {a b : α}
    (h : (⟦a⟧ : orbitRel.Quotient H α) = ⟦b⟧) : (⟦a⟧ : orbitRel.Quotient G α) = ⟦b⟧ := by
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    H : Subgroup G
    a b : α
    h : Eq (Quotient.mk (MulAction.orbitRel (Subtype fun x => Membership.mem H x)  …
    ⊢ Eq (Quotient.mk (MulAction.orbitRel G α) a) (Quotient.mk (MulAction.orbitRel …
  -/
  rw [@Quotient.eq] at h ⊢
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    H : Subgroup G
    a b : α
    h : (MulAction.orbitRel (Subtype fun x => Membership.mem H x) α) a b
    ⊢ (MulAction.orbitRel G α) a b
  -/
  exact mem_orbit_of_mem_orbit_subgroup h
  /-
    🎉 no goals
  -/


@[to_additive]
lemma orbitRel.quotient_eq_of_quotient_subgroup_eq' {H : Subgroup G} {a b : α}
    (h : (Quotient.mk'' a : orbitRel.Quotient H α) = Quotient.mk'' b) :
    (Quotient.mk'' a : orbitRel.Quotient G α) = Quotient.mk'' b :=
  orbitRel.quotient_eq_of_quotient_subgroup_eq h


@[to_additive]
nonrec lemma orbitRel.Quotient.orbit_nonempty (x : orbitRel.Quotient G α) :
    Set.Nonempty x.orbit := by
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    x : MulAction.orbitRel.Quotient G α
    ⊢ x.orbit.Nonempty
  -/
  rw [orbitRel.Quotient.orbit_eq_orbit_out x Quotient.out_eq']
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    x : MulAction.orbitRel.Quotient G α
    ⊢ (MulAction.orbit G (Quotient.out x)).Nonempty
  -/
  exact orbit_nonempty _
  /-
    🎉 no goals
  -/


@[to_additive]
nonrec lemma orbitRel.Quotient.mapsTo_smul_orbit (g : G) (x : orbitRel.Quotient G α) :
    Set.MapsTo (g • ·) x.orbit x.orbit := by
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    g : G
    x : MulAction.orbitRel.Quotient G α
    ⊢ Set.MapsTo (fun x => HSMul.hSMul g x) x.orbit x.orbit
  -/
  rw [orbitRel.Quotient.orbit_eq_orbit_out x Quotient.out_eq']
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    g : G
    x : MulAction.orbitRel.Quotient G α
    ⊢ Set.MapsTo (fun x => HSMul.hSMul g x) (MulAction.orbit G (Quotient.out x)) ( …
  -/
  exact mapsTo_smul_orbit g x.out
  /-
    🎉 no goals
  -/


@[to_additive]
instance (x : orbitRel.Quotient G α) : MulAction G x.orbit where
  smul g := (orbitRel.Quotient.mapsTo_smul_orbit g x).restrict _ _ _
  one_smul a := Subtype.ext (one_smul G (a : α))
  mul_smul g g' a' := Subtype.ext (mul_smul g g' (a' : α))


@[to_additive (attr := simp)]
lemma orbitRel.Quotient.orbit.coe_smul {g : G} {x : orbitRel.Quotient G α} {a : x.orbit} :
    ↑(g • a) = g • (a : α) :=
  rfl


@[to_additive (attr := norm_cast, simp)]
lemma orbitRel.Quotient.mem_subgroup_orbit_iff {H : Subgroup G} {x : orbitRel.Quotient G α}
    {a b : x.orbit} : (a : α) ∈ MulAction.orbit H (b : α) ↔ a ∈ MulAction.orbit H b := by
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    H : Subgroup G
    x : MulAction.orbitRel.Quotient G α
    a b : ↑x.orbit
    ⊢ Iff (Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem H x) ↑ …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ?_⟩
    /-
      case refine_1
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      H : Subgroup G
      x : MulAction.orbitRel.Quotient G α
      a b : ↑x.orbit
      h : Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem H x) ↑b) ↑a
      ⊢ Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem H x) b) a
    -/
  · rcases h with ⟨g, h⟩
    /-
      case refine_1.intro
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      H : Subgroup G
      x : MulAction.orbitRel.Quotient G α
      a b : ↑x.orbit
      g : Subtype fun x => Membership.mem H x
      h : Eq ((fun m => HSMul.hSMul m ↑b) g) ↑a
      ⊢ Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem H x) b) a
    -/
    dsimp at h
    /-
      case refine_1.intro
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      H : Subgroup G
      x : MulAction.orbitRel.Quotient G α
      a b : ↑x.orbit
      g : Subtype fun x => Membership.mem H x
      h : Eq (HSMul.hSMul g ↑b) ↑a
      ⊢ Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem H x) b) a
    -/
    erw [← orbit.coe_smul, ← Subtype.ext_iff] at h
    /-
      case refine_1.intro
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      H : Subgroup G
      x : MulAction.orbitRel.Quotient G α
      a b : ↑x.orbit
      g : Subtype fun x => Membership.mem H x
      h : Eq (HSMul.hSMul (H.subtype g) b) a
      ⊢ Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem H x) b) a
    -/
    subst h
    /-
      case refine_1.intro
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      H : Subgroup G
      x : MulAction.orbitRel.Quotient G α
      b : ↑x.orbit
      g : Subtype fun x => Membership.mem H x
      ⊢ Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem H x) b) (HS …
    -/
    exact MulAction.mem_orbit _ g
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      H : Subgroup G
      x : MulAction.orbitRel.Quotient G α
      a b : ↑x.orbit
      h : Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem H x) b) a
      ⊢ Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem H x) ↑b) ↑a
    -/
  · rcases h with ⟨g, rfl⟩
    /-
      case refine_2.intro
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      H : Subgroup G
      x : MulAction.orbitRel.Quotient G α
      b : ↑x.orbit
      g : Subtype fun x => Membership.mem H x
      ⊢ Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem H x) ↑b) ↑( …
    -/
    exact MulAction.mem_orbit _ g
    /-
      🎉 no goals
    -/


@[to_additive]
lemma orbitRel.Quotient.subgroup_quotient_eq_iff {H : Subgroup G} {x : orbitRel.Quotient G α}
    {a b : x.orbit} : (⟦a⟧ : orbitRel.Quotient H x.orbit) = ⟦b⟧ ↔
      (⟦↑a⟧ : orbitRel.Quotient H α) = ⟦↑b⟧ := by
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    H : Subgroup G
    x : MulAction.orbitRel.Quotient G α
    a b : ↑x.orbit
    ⊢ Iff (Eq (Quotient.mk (MulAction.orbitRel (Subtype fun x => Membership.mem H  …
  -/
  simp_rw [← @Quotient.mk''_eq_mk, Quotient.eq'']
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    H : Subgroup G
    x : MulAction.orbitRel.Quotient G α
    a b : ↑x.orbit
    ⊢ Iff ((MulAction.orbitRel (Subtype fun x => Membership.mem H x) ↑x.orbit) a b …
  -/
  exact orbitRel.Quotient.mem_subgroup_orbit_iff.symm
  /-
    🎉 no goals
  -/


@[to_additive]
lemma orbitRel.Quotient.mem_subgroup_orbit_iff' {H : Subgroup G} {x : orbitRel.Quotient G α}
    {a b : x.orbit} {c : α} (h : (⟦a⟧ : orbitRel.Quotient H x.orbit) = ⟦b⟧) :
    (a : α) ∈ MulAction.orbit H c ↔ (b : α) ∈ MulAction.orbit H c := by
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    H : Subgroup G
    x : MulAction.orbitRel.Quotient G α
    a b : ↑x.orbit
    c : α
    h : Eq (Quotient.mk (MulAction.orbitRel (Subtype fun x => Membership.mem H x)  …
    ⊢ Iff (Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem H x) c …
  -/
  simp_rw [mem_orbit_symm (a₂ := c)]
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    H : Subgroup G
    x : MulAction.orbitRel.Quotient G α
    a b : ↑x.orbit
    c : α
    h : Eq (Quotient.mk (MulAction.orbitRel (Subtype fun x => Membership.mem H x)  …
    ⊢ Iff (Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem H x) ↑ …
  -/
  convert Iff.rfl using 2
  /-
    case h.e'_2.h.e'_4
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    H : Subgroup G
    x : MulAction.orbitRel.Quotient G α
    a b : ↑x.orbit
    c : α
    h : Eq (Quotient.mk (MulAction.orbitRel (Subtype fun x => Membership.mem H x)  …
    ⊢ Eq (MulAction.orbit (Subtype fun x => Membership.mem H x) ↑b) (MulAction.orb …
  -/
  rw [orbit_eq_iff]
  suffices hb : ↑b ∈ orbitRel.Quotient.orbit (⟦a⟧ : orbitRel.Quotient H x.orbit) by
    rw [orbitRel.Quotient.orbit_eq_orbit_out (⟦a⟧ : orbitRel.Quotient H x.orbit) Quotient.out_eq']
       at hb
    rw [orbitRel.Quotient.mem_subgroup_orbit_iff]
    convert hb using 1
    rw [orbit_eq_iff, ← orbitRel_apply, ← Quotient.eq'', Quotient.out_eq', @Quotient.mk''_eq_mk]
  /-
    case h.e'_2.h.e'_4
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    H : Subgroup G
    x : MulAction.orbitRel.Quotient G α
    a b : ↑x.orbit
    c : α
    h : Eq (Quotient.mk (MulAction.orbitRel (Subtype fun x => Membership.mem H x)  …
    ⊢ Membership.mem (MulAction.orbitRel.Quotient.orbit (Quotient.mk (MulAction.or …
  -/
  rw [orbitRel.Quotient.mem_orbit, h, @Quotient.mk''_eq_mk]
  /-
    🎉 no goals
  -/


local notation "Ω" => orbitRel.Quotient G α


/-- Decomposition of a type `X` as a disjoint union of its orbits under a group action.

This version is expressed in terms of `MulAction.orbitRel.Quotient.orbit` instead of
`MulAction.orbit`, to avoid mentioning `Quotient.out`. -/
@[to_additive
      "Decomposition of a type `X` as a disjoint union of its orbits under an additive group action.

      This version is expressed in terms of `AddAction.orbitRel.Quotient.orbit` instead of
      `AddAction.orbit`, to avoid mentioning `Quotient.out`. "]
def selfEquivSigmaOrbits' : α ≃ Σω : Ω, ω.orbit :=
  letI := orbitRel G α
  calc
    α ≃ Σω : Ω, { a // Quotient.mk' a = ω } := (Equiv.sigmaFiberEquiv Quotient.mk').symm
    _ ≃ Σω : Ω, ω.orbit :=
      Equiv.sigmaCongrRight fun _ =>
        Equiv.subtypeEquivRight fun _ => orbitRel.Quotient.mem_orbit.symm


/-- Decomposition of a type `X` as a disjoint union of its orbits under a group action. -/
@[to_additive
      "Decomposition of a type `X` as a disjoint union of its orbits under an additive group
      action."]
def selfEquivSigmaOrbits : α ≃ Σω : Ω, orbit G ω.out :=
  (selfEquivSigmaOrbits' G α).trans <|
    Equiv.sigmaCongrRight fun _ =>
      Equiv.Set.ofEq <| orbitRel.Quotient.orbit_eq_orbit_out _ Quotient.out_eq'


/-- Decomposition of a type `X` as a disjoint union of its orbits under a group action.
Phrased as a set union. See `MulAction.selfEquivSigmaOrbits` for the type isomorphism. -/
@[to_additive "Decomposition of a type `X` as a disjoint union of its orbits under an additive group
action. Phrased as a set union. See `AddAction.selfEquivSigmaOrbits` for the type isomorphism."]
lemma univ_eq_iUnion_orbit :
    Set.univ (α := α) = ⋃ x : Ω, x.orbit := by
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    ⊢ Eq Set.univ (Set.iUnion fun x => x.orbit)
  -/
  ext x
  /-
    case h
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    x : α
    ⊢ Iff (Membership.mem Set.univ x) (Membership.mem (Set.iUnion fun x => x.orbit …
  -/
  simp only [Set.mem_univ, Set.mem_iUnion, true_iff]
  /-
    case h
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    x : α
    ⊢ Exists fun i => Membership.mem i.orbit x
  -/
  exact ⟨Quotient.mk'' x, by simp⟩
  /-
    🎉 no goals
  -/


/-- The stabilizer of an element under an action, i.e. what sends the element to itself.
A subgroup. -/
@[to_additive
      "The stabilizer of an element under an action, i.e. what sends the element to itself.
      An additive subgroup."]
def stabilizer (a : α) : Subgroup G :=
  { stabilizerSubmonoid G a with
                                                                /-
                                                                  G : Type u_1
                                                                  α : Type u_2
                                                                  β : Type u_3
                                                                  inst✝² : Group G
                                                                  inst✝¹ : MulAction G α
                                                                  inst✝ : MulAction G β
                                                                  a : α
                                                                  m : G
                                                                  ha : Eq (HSMul.hSMul m a) a
                                                                  ⊢ Eq (HSMul.hSMul (Inv.inv m) a) a
                                                                -/
    inv_mem' := fun {m} (ha : m • a = a) => show m⁻¹ • a = a by rw [inv_smul_eq_iff, ha] }
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[to_additive]
instance [DecidableEq α] (a : α) : DecidablePred (· ∈ stabilizer G a) :=
  fun _ => inferInstanceAs <| Decidable (_ = _)


@[to_additive (attr := simp)]
theorem mem_stabilizer_iff {a : α} {g : G} : g ∈ stabilizer G a ↔ g • a = a :=
  Iff.rfl


@[to_additive]
lemma le_stabilizer_smul_left [SMul α β] [IsScalarTower G α β] (a : α) (b : β) :
    stabilizer G a ≤ stabilizer G (a • b) := by
  /-
    G : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝⁴ : Group G
    inst✝³ : MulAction G α
    inst✝² : MulAction G β
    inst✝¹ : SMul α β
    inst✝ : IsScalarTower G α β
    a : α
    b : β
    ⊢ LE.le (MulAction.stabilizer G a) (MulAction.stabilizer G (HSMul.hSMul a b))
  -/
  simp_rw [SetLike.le_def, mem_stabilizer_iff, ← smul_assoc]; rintro a h; rw [h]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/

-- This lemma does not need `MulAction G α`, only `SMul G α`.
-- We use `G'` instead of `G` to locally reduce the typeclass assumptions.

@[to_additive]
lemma le_stabilizer_smul_right {G'} [Group G'] [SMul α β] [MulAction G' β]
    [SMulCommClass G' α β] (a : α) (b : β) :
    stabilizer G' b ≤ stabilizer G' (a • b) := by
  /-
    α : Type u_2
    β : Type u_3
    G' : Type u_4
    inst✝³ : Group G'
    inst✝² : SMul α β
    inst✝¹ : MulAction G' β
    inst✝ : SMulCommClass G' α β
    a : α
    b : β
    ⊢ LE.le (MulAction.stabilizer G' b) (MulAction.stabilizer G' (HSMul.hSMul a b))
  -/
  simp_rw [SetLike.le_def, mem_stabilizer_iff, smul_comm]; rintro a h; rw [h]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[to_additive (attr := simp)]
lemma stabilizer_smul_eq_left [SMul α β] [IsScalarTower G α β] (a : α) (b : β)
    (h : Injective (· • b : α → β)) : stabilizer G (a • b) = stabilizer G a := by
  /-
    G : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝⁴ : Group G
    inst✝³ : MulAction G α
    inst✝² : MulAction G β
    inst✝¹ : SMul α β
    inst✝ : IsScalarTower G α β
    a : α
    b : β
    h : Function.Injective fun x => HSMul.hSMul x b
    ⊢ Eq (MulAction.stabilizer G (HSMul.hSMul a b)) (MulAction.stabilizer G a)
  -/
  refine (le_stabilizer_smul_left _ _).antisymm' fun a ha ↦ ?_
  /-
    G : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝⁴ : Group G
    inst✝³ : MulAction G α
    inst✝² : MulAction G β
    inst✝¹ : SMul α β
    inst✝ : IsScalarTower G α β
    a✝ : α
    b : β
    h : Function.Injective fun x => HSMul.hSMul x b
    a : G
    ha : Membership.mem (MulAction.stabilizer G (HSMul.hSMul a✝ b)) a
    ⊢ Membership.mem (MulAction.stabilizer G a✝) a
  -/
  simpa only [mem_stabilizer_iff, ← smul_assoc, h.eq_iff] using ha
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma stabilizer_smul_eq_right {α} [Group α] [MulAction α β] [SMulCommClass G α β] (a : α) (b : β) :
    stabilizer G (a • b) = stabilizer G b :=
  (le_stabilizer_smul_right _ _).antisymm' <| (le_stabilizer_smul_right a⁻¹ _).trans_eq <| by
    /-
      G : Type u_1
      β : Type u_3
      inst✝⁴ : Group G
      inst✝³ : MulAction G β
      α : Type u_4
      inst✝² : Group α
      inst✝¹ : MulAction α β
      inst✝ : SMulCommClass G α β
      a : α
      b : β
      ⊢ Eq (MulAction.stabilizer G (HSMul.hSMul (Inv.inv a) (HSMul.hSMul a b))) (Mul …
    -/
    rw [inv_smul_smul]
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp)]
lemma stabilizer_mul_eq_left [Group α] [IsScalarTower G α α] (a b : α)  :
    stabilizer G (a * b) = stabilizer G a := stabilizer_smul_eq_left a _ <| mul_left_injective _


@[to_additive (attr := simp)]
lemma stabilizer_mul_eq_right [Group α] [SMulCommClass G α α] (a b : α) :
    stabilizer G (a * b) = stabilizer G b := stabilizer_smul_eq_right a _


