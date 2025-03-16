@[to_additive]
lemma fst_mem_orbit_of_mem_orbit {x y : α × β} (h : x ∈ MulAction.orbit M y) :
    x.1 ∈ MulAction.orbit M y.1 := by
  /-
    M : Type u
    inst✝² : Monoid M
    α : Type v
    inst✝¹ : MulAction M α
    β : Type u_1
    inst✝ : MulAction M β
    x y : Prod α β
    h : Membership.mem (MulAction.orbit M y) x
    ⊢ Membership.mem (MulAction.orbit M y.1) x.1
  -/
  rcases h with ⟨g, rfl⟩
  /-
    case intro
    M : Type u
    inst✝² : Monoid M
    α : Type v
    inst✝¹ : MulAction M α
    β : Type u_1
    inst✝ : MulAction M β
    y : Prod α β
    g : M
    ⊢ Membership.mem (MulAction.orbit M y.1) ((fun m => HSMul.hSMul m y) g).1
  -/
  exact mem_orbit _ _
  /-
    🎉 no goals
  -/


@[to_additive]
lemma snd_mem_orbit_of_mem_orbit {x y : α × β} (h : x ∈ MulAction.orbit M y) :
    x.2 ∈ MulAction.orbit M y.2 := by
  /-
    M : Type u
    inst✝² : Monoid M
    α : Type v
    inst✝¹ : MulAction M α
    β : Type u_1
    inst✝ : MulAction M β
    x y : Prod α β
    h : Membership.mem (MulAction.orbit M y) x
    ⊢ Membership.mem (MulAction.orbit M y.2) x.2
  -/
  rcases h with ⟨g, rfl⟩
  /-
    case intro
    M : Type u
    inst✝² : Monoid M
    α : Type v
    inst✝¹ : MulAction M α
    β : Type u_1
    inst✝ : MulAction M β
    y : Prod α β
    g : M
    ⊢ Membership.mem (MulAction.orbit M y.2) ((fun m => HSMul.hSMul m y) g).2
  -/
  exact mem_orbit _ _
  /-
    🎉 no goals
  -/


@[to_additive]
lemma _root_.Finite.finite_mulAction_orbit [Finite M] (a : α) : Set.Finite (orbit M a) :=
  Set.finite_range _


@[to_additive]
theorem orbit_eq_univ [IsPretransitive M α] (a : α) : orbit M a = Set.univ :=
  (surjective_smul M a).range_eq


@[to_additive mem_fixedPoints_iff_card_orbit_eq_one]
theorem mem_fixedPoints_iff_card_orbit_eq_one {a : α} [Fintype (orbit M a)] :
    a ∈ fixedPoints M α ↔ Fintype.card (orbit M a) = 1 := by
  /-
    M : Type u
    inst✝² : Monoid M
    α : Type v
    inst✝¹ : MulAction M α
    a : α
    inst✝ : Fintype ↑(MulAction.orbit M a)
    ⊢ Iff (Membership.mem (MulAction.fixedPoints M α) a) (Eq (Fintype.card ↑(MulAc …
  -/
  rw [Fintype.card_eq_one_iff, mem_fixedPoints]
  /-
    M : Type u
    inst✝² : Monoid M
    α : Type v
    inst✝¹ : MulAction M α
    a : α
    inst✝ : Fintype ↑(MulAction.orbit M a)
    ⊢ Iff (∀ (m : M), Eq (HSMul.hSMul m a) a) (Exists fun x => ∀ (y : ↑(MulAction. …
  -/
  constructor
    /-
      case mp
      M : Type u
      inst✝² : Monoid M
      α : Type v
      inst✝¹ : MulAction M α
      a : α
      inst✝ : Fintype ↑(MulAction.orbit M a)
      ⊢ (∀ (m : M), Eq (HSMul.hSMul m a) a) → Exists fun x => ∀ (y : ↑(MulAction.orb …
    -/
  · exact fun h => ⟨⟨a, mem_orbit_self _⟩, fun ⟨a, ⟨x, hx⟩⟩ => Subtype.eq <| by simp [h x, hx.symm]⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      M : Type u
      inst✝² : Monoid M
      α : Type v
      inst✝¹ : MulAction M α
      a : α
      inst✝ : Fintype ↑(MulAction.orbit M a)
      ⊢ (Exists fun x => ∀ (y : ↑(MulAction.orbit M a)), Eq y x) → ∀ (m : M), Eq (HS …
    -/
  · intro h x
    /-
      case mpr
      M : Type u
      inst✝² : Monoid M
      α : Type v
      inst✝¹ : MulAction M α
      a : α
      inst✝ : Fintype ↑(MulAction.orbit M a)
      h : Exists fun x => ∀ (y : ↑(MulAction.orbit M a)), Eq y x
      x : M
      ⊢ Eq (HSMul.hSMul x a) a
    -/
    rcases h with ⟨⟨z, hz⟩, hz₁⟩
    calc
      x • a = z := Subtype.mk.inj (hz₁ ⟨x • a, mem_orbit _ _⟩)
      _ = a := (Subtype.mk.inj (hz₁ ⟨a, mem_orbit_self _⟩)).symm


@[to_additive instDecidablePredMemSetFixedByAddOfDecidableEq]
instance (m : M) [DecidableEq β] :
    DecidablePred fun b : β => b ∈ MulAction.fixedBy β m := fun b ↦ by
  /-
    M : Type u
    inst✝³ : Monoid M
    α : Type v
    inst✝² : MulAction M α
    β : Type u_1
    inst✝¹ : MulAction M β
    m : M
    inst✝ : DecidableEq β
    b : β
    ⊢ Decidable ((fun b => Membership.mem (MulAction.fixedBy β m) b) b)
  -/
  simp only [MulAction.mem_fixedBy, Equiv.Perm.smul_def]
  /-
    M : Type u
    inst✝³ : Monoid M
    α : Type v
    inst✝² : MulAction M α
    β : Type u_1
    inst✝¹ : MulAction M β
    m : M
    inst✝ : DecidableEq β
    b : β
    ⊢ Decidable (Eq (HSMul.hSMul m b) b)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- `smul` by a `k : M` over a ring is injective, if `k` is not a zero divisor.
The general theory of such `k` is elaborated by `IsSMulRegular`.
The typeclass that restricts all terms of `M` to have this property is `NoZeroSMulDivisors`. -/
theorem smul_cancel_of_non_zero_divisor {M R : Type*} [Monoid M] [NonUnitalNonAssocRing R]
    [DistribMulAction M R] (k : M) (h : ∀ x : R, k • x = 0 → x = 0) {a b : R} (h' : k • a = k • b) :
    a = b := by
  /-
    M : Type u_1
    R : Type u_2
    inst✝² : Monoid M
    inst✝¹ : NonUnitalNonAssocRing R
    inst✝ : DistribMulAction M R
    k : M
    h : ∀ (x : R), Eq (HSMul.hSMul k x) 0 → Eq x 0
    a b : R
    h' : Eq (HSMul.hSMul k a) (HSMul.hSMul k b)
    ⊢ Eq a b
  -/
  rw [← sub_eq_zero]
  /-
    M : Type u_1
    R : Type u_2
    inst✝² : Monoid M
    inst✝¹ : NonUnitalNonAssocRing R
    inst✝ : DistribMulAction M R
    k : M
    h : ∀ (x : R), Eq (HSMul.hSMul k x) 0 → Eq x 0
    a b : R
    h' : Eq (HSMul.hSMul k a) (HSMul.hSMul k b)
    ⊢ Eq (HSub.hSub a b) 0
  -/
  refine h _ ?_
  /-
    M : Type u_1
    R : Type u_2
    inst✝² : Monoid M
    inst✝¹ : NonUnitalNonAssocRing R
    inst✝ : DistribMulAction M R
    k : M
    h : ∀ (x : R), Eq (HSMul.hSMul k x) 0 → Eq x 0
    a b : R
    h' : Eq (HSMul.hSMul k a) (HSMul.hSMul k b)
    ⊢ Eq (HSMul.hSMul k (HSub.hSub a b)) 0
  -/
  rw [smul_sub, h', sub_self]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem smul_orbit (g : G) (a : α) : g • orbit G a = orbit G a :=
  (smul_orbit_subset g a).antisymm <|
    calc
      orbit G a = g • g⁻¹ • orbit G a := (smul_inv_smul _ _).symm
      _ ⊆ g • orbit G a := Set.image_subset _ (smul_orbit_subset _ _)


/-- The action of a group on an orbit is transitive. -/
@[to_additive "The action of an additive group on an orbit is transitive."]
instance (a : α) : IsPretransitive G (orbit G a) :=
  ⟨by
    /-
      G : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝² : Group G
      inst✝¹ : MulAction G α
      inst✝ : MulAction G β
      a : α
      ⊢ ∀ (x y : ↑(MulAction.orbit G a)), Exists fun g => Eq (HSMul.hSMul g x) y
    -/
    rintro ⟨_, g, rfl⟩ ⟨_, h, rfl⟩
    /-
      case mk.intro.mk.intro
      G : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝² : Group G
      inst✝¹ : MulAction G α
      inst✝ : MulAction G β
      a : α
      g h : G
      ⊢ Exists fun g_1 => Eq (HSMul.hSMul g_1 ⟨(fun m => HSMul.hSMul m a) g, ⋯⟩) ⟨(f …
    -/
    use h * g⁻¹
    /-
      case h
      G : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝² : Group G
      inst✝¹ : MulAction G α
      inst✝ : MulAction G β
      a : α
      g h : G
      ⊢ Eq (HSMul.hSMul (HMul.hMul h (Inv.inv g)) ⟨(fun m => HSMul.hSMul m a) g, ⋯⟩) …
    -/
    ext1
    /-
      case h.a
      G : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝² : Group G
      inst✝¹ : MulAction G α
      inst✝ : MulAction G β
      a : α
      g h : G
      ⊢ Eq ↑(HSMul.hSMul (HMul.hMul h (Inv.inv g)) ⟨(fun m => HSMul.hSMul m a) g, ⋯⟩ …
    -/
    simp [mul_smul]⟩
    /-
      🎉 no goals
    -/


@[to_additive]
lemma orbitRel_subgroup_le (H : Subgroup G) : orbitRel H α ≤ orbitRel G α :=
  Setoid.le_def.2 mem_orbit_of_mem_orbit_subgroup


@[to_additive]
lemma orbitRel_subgroupOf (H K : Subgroup G) :
    orbitRel (H.subgroupOf K) α = orbitRel (H ⊓ K : Subgroup G) α := by
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    H K : Subgroup G
    ⊢ Eq (MulAction.orbitRel (Subtype fun x => Membership.mem (H.subgroupOf K) x)  …
  -/
  rw [← Subgroup.subgroupOf_map_subtype]
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    H K : Subgroup G
    ⊢ Eq (MulAction.orbitRel (Subtype fun x => Membership.mem (H.subgroupOf K) x)  …
  -/
  ext x
  /-
    case a
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    H K : Subgroup G
    x b✝ : α
    ⊢ Iff ((MulAction.orbitRel (Subtype fun x => Membership.mem (H.subgroupOf K) x …
  -/
  simp_rw [orbitRel_apply]
  /-
    case a
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    H K : Subgroup G
    x b✝ : α
    ⊢ Iff (Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem (H.sub …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ?_⟩
    /-
      case a.refine_1
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      H K : Subgroup G
      x b✝ : α
      h : Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem (H.subgro …
      ⊢ Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem (Subgroup.m …
    -/
  · rcases h with ⟨⟨gv, gp⟩, rfl⟩
    /-
      case a.refine_1.intro.mk
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      H K : Subgroup G
      b✝ : α
      gv : Subtype fun x => Membership.mem K x
      gp : Membership.mem (H.subgroupOf K) gv
      ⊢ Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem (Subgroup.m …
    -/
    simp only [Submonoid.mk_smul]
    /-
      case a.refine_1.intro.mk
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      H K : Subgroup G
      b✝ : α
      gv : Subtype fun x => Membership.mem K x
      gp : Membership.mem (H.subgroupOf K) gv
      ⊢ Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem (Subgroup.m …
    -/
    refine mem_orbit _ (⟨gv, ?_⟩ : Subgroup.map K.subtype (H.subgroupOf K))
    /-
      case a.refine_1.intro.mk
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      H K : Subgroup G
      b✝ : α
      gv : Subtype fun x => Membership.mem K x
      gp : Membership.mem (H.subgroupOf K) gv
      ⊢ Membership.mem (Subgroup.map K.subtype (H.subgroupOf K)) ↑gv
    -/
    simpa using gp
    /-
      🎉 no goals
    -/
    /-
      case a.refine_2
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      H K : Subgroup G
      x b✝ : α
      h : Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem (Subgroup …
      ⊢ Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem (H.subgroup …
    -/
  · rcases h with ⟨⟨gv, gp⟩, rfl⟩
    /-
      case a.refine_2.intro.mk
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      H K : Subgroup G
      b✝ : α
      gv : G
      gp : Membership.mem (Subgroup.map K.subtype (H.subgroupOf K)) gv
      ⊢ Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem (H.subgroup …
    -/
    simp only [Submonoid.mk_smul]
    /-
      case a.refine_2.intro.mk
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      H K : Subgroup G
      b✝ : α
      gv : G
      gp : Membership.mem (Subgroup.map K.subtype (H.subgroupOf K)) gv
      ⊢ Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem (H.subgroup …
    -/
    simp only [Subgroup.subgroupOf_map_subtype, Subgroup.mem_inf] at gp
    /-
      case a.refine_2.intro.mk
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      H K : Subgroup G
      b✝ : α
      gv : G
      gp✝ : Membership.mem (Subgroup.map K.subtype (H.subgroupOf K)) gv
      gp : And (Membership.mem H gv) (Membership.mem K gv)
      ⊢ Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem (H.subgroup …
    -/
    refine mem_orbit _ (⟨⟨gv, ?_⟩, ?_⟩ : H.subgroupOf K)
      /-
        case a.refine_2.intro.mk.refine_1
        G : Type u_1
        α : Type u_2
        inst✝¹ : Group G
        inst✝ : MulAction G α
        H K : Subgroup G
        b✝ : α
        gv : G
        gp✝ : Membership.mem (Subgroup.map K.subtype (H.subgroupOf K)) gv
        gp : And (Membership.mem H gv) (Membership.mem K gv)
        ⊢ Membership.mem K gv
      -/
    · exact gp.2
      /-
        🎉 no goals
      -/
      /-
        case a.refine_2.intro.mk.refine_2
        G : Type u_1
        α : Type u_2
        inst✝¹ : Group G
        inst✝ : MulAction G α
        H K : Subgroup G
        b✝ : α
        gv : G
        gp✝ : Membership.mem (Subgroup.map K.subtype (H.subgroupOf K)) gv
        gp : And (Membership.mem H gv) (Membership.mem K gv)
        ⊢ Membership.mem (H.subgroupOf K) ⟨gv, ⋯⟩
      -/
    · simp only [Subgroup.mem_subgroupOf]
      /-
        case a.refine_2.intro.mk.refine_2
        G : Type u_1
        α : Type u_2
        inst✝¹ : Group G
        inst✝ : MulAction G α
        H K : Subgroup G
        b✝ : α
        gv : G
        gp✝ : Membership.mem (Subgroup.map K.subtype (H.subgroupOf K)) gv
        gp : And (Membership.mem H gv) (Membership.mem K gv)
        ⊢ Membership.mem H gv
      -/
      exact gp.1
      /-
        🎉 no goals
      -/


/-- An action is pretransitive if and only if the quotient by `MulAction.orbitRel` is a
subsingleton. -/
@[to_additive "An additive action is pretransitive if and only if the quotient by
`AddAction.orbitRel` is a subsingleton."]
theorem pretransitive_iff_subsingleton_quotient :
    IsPretransitive G α ↔ Subsingleton (orbitRel.Quotient G α) := by
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    ⊢ Iff (MulAction.IsPretransitive G α) (Subsingleton (MulAction.orbitRel.Quotie …
  -/
  refine ⟨fun _ ↦ ⟨fun a b ↦ ?_⟩, fun _ ↦ ⟨fun a b ↦ ?_⟩⟩
    /-
      case refine_1
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      x✝ : MulAction.IsPretransitive G α
      a b : MulAction.orbitRel.Quotient G α
      ⊢ Eq a b
    -/
  · refine Quot.inductionOn a (fun x ↦ ?_)
    /-
      case refine_1
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      x✝ : MulAction.IsPretransitive G α
      a b : MulAction.orbitRel.Quotient G α
      x : α
      ⊢ Eq (Quot.mk (⇑(MulAction.orbitRel G α)) x) b
    -/
    exact Quot.inductionOn b (fun y ↦ Quot.sound <| exists_smul_eq G y x)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      x✝ : Subsingleton (MulAction.orbitRel.Quotient G α)
      a b : α
      ⊢ Exists fun g => Eq (HSMul.hSMul g a) b
    -/
  · have h : Quotient.mk (orbitRel G α) b = ⟦a⟧ := Subsingleton.elim _ _
    /-
      case refine_2
      G : Type u_1
      α : Type u_2
      inst✝¹ : Group G
      inst✝ : MulAction G α
      x✝ : Subsingleton (MulAction.orbitRel.Quotient G α)
      a b : α
      h : Eq (Quotient.mk (MulAction.orbitRel G α) b) (Quotient.mk (MulAction.orbitR …
      ⊢ Exists fun g => Eq (HSMul.hSMul g a) b
    -/
    exact Quotient.eq''.mp h
    /-
      🎉 no goals
    -/


/-- If `α` is non-empty, an action is pretransitive if and only if the quotient has exactly one
element. -/
@[to_additive "If `α` is non-empty, an additive action is pretransitive if and only if the
quotient has exactly one element."]
theorem pretransitive_iff_unique_quotient_of_nonempty [Nonempty α] :
    IsPretransitive G α ↔ Nonempty (Unique <| orbitRel.Quotient G α) := by
  /-
    G : Type u_1
    α : Type u_2
    inst✝² : Group G
    inst✝¹ : MulAction G α
    inst✝ : Nonempty α
    ⊢ Iff (MulAction.IsPretransitive G α) (Nonempty (Unique (MulAction.orbitRel.Qu …
  -/
  rw [unique_iff_subsingleton_and_nonempty, pretransitive_iff_subsingleton_quotient, iff_self_and]
  /-
    G : Type u_1
    α : Type u_2
    inst✝² : Group G
    inst✝¹ : MulAction G α
    inst✝ : Nonempty α
    ⊢ Subsingleton (MulAction.orbitRel.Quotient G α) → Nonempty (MulAction.orbitRe …
  -/
  exact fun _ ↦ (nonempty_quotient_iff _).mpr inferInstance
  /-
    🎉 no goals
  -/


@[to_additive]
instance (x : orbitRel.Quotient G α) : IsPretransitive G x.orbit where
  exists_smul_eq := by
    /-
      G : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝² : Group G
      inst✝¹ : MulAction G α
      inst✝ : MulAction G β
      x : MulAction.orbitRel.Quotient G α
      ⊢ ∀ (x_1 y : ↑x.orbit), Exists fun g => Eq (HSMul.hSMul g x_1) y
    -/
    induction x using Quotient.inductionOn'
    /-
      case h
      G : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝² : Group G
      inst✝¹ : MulAction G α
      inst✝ : MulAction G β
      a✝ : α
      ⊢ ∀ (x y : ↑(MulAction.orbitRel.Quotient.orbit (Quotient.mk'' a✝))), Exists fu …
    -/
    rintro ⟨y, yh⟩ ⟨z, zh⟩
    /-
      case h.mk.mk
      G : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝² : Group G
      inst✝¹ : MulAction G α
      inst✝ : MulAction G β
      a✝ y : α
      yh : Membership.mem (MulAction.orbitRel.Quotient.orbit (Quotient.mk'' a✝)) y
      z : α
      zh : Membership.mem (MulAction.orbitRel.Quotient.orbit (Quotient.mk'' a✝)) z
      ⊢ Exists fun g => Eq (HSMul.hSMul g ⟨y, yh⟩) ⟨z, zh⟩
    -/
    rw [orbitRel.Quotient.mem_orbit, Quotient.eq''] at yh zh
    /-
      case h.mk.mk
      G : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝² : Group G
      inst✝¹ : MulAction G α
      inst✝ : MulAction G β
      a✝ y : α
      yh✝ : Membership.mem (MulAction.orbitRel.Quotient.orbit (Quotient.mk'' a✝)) y
      yh : (MulAction.orbitRel G α) y a✝
      z : α
      zh✝ : Membership.mem (MulAction.orbitRel.Quotient.orbit (Quotient.mk'' a✝)) z
      zh : (MulAction.orbitRel G α) z a✝
      ⊢ Exists fun g => Eq (HSMul.hSMul g ⟨y, yh✝⟩) ⟨z, zh✝⟩
    -/
    rcases yh with ⟨g, rfl⟩
    /-
      case h.mk.mk.intro
      G : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝² : Group G
      inst✝¹ : MulAction G α
      inst✝ : MulAction G β
      a✝ z : α
      zh✝ : Membership.mem (MulAction.orbitRel.Quotient.orbit (Quotient.mk'' a✝)) z
      zh : (MulAction.orbitRel G α) z a✝
      g : G
      yh : Membership.mem (MulAction.orbitRel.Quotient.orbit (Quotient.mk'' a✝)) ((f …
      ⊢ Exists fun g_1 => Eq (HSMul.hSMul g_1 ⟨(fun m => HSMul.hSMul m a✝) g, yh⟩) ⟨ …
    -/
    rcases zh with ⟨h, rfl⟩
    /-
      case h.mk.mk.intro.intro
      G : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝² : Group G
      inst✝¹ : MulAction G α
      inst✝ : MulAction G β
      a✝ : α
      g : G
      yh : Membership.mem (MulAction.orbitRel.Quotient.orbit (Quotient.mk'' a✝)) ((f …
      h : G
      zh : Membership.mem (MulAction.orbitRel.Quotient.orbit (Quotient.mk'' a✝)) ((f …
      ⊢ Exists fun g_1 => Eq (HSMul.hSMul g_1 ⟨(fun m => HSMul.hSMul m a✝) g, yh⟩) ⟨ …
    -/
    refine ⟨h * g⁻¹, ?_⟩
    /-
      case h.mk.mk.intro.intro
      G : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝² : Group G
      inst✝¹ : MulAction G α
      inst✝ : MulAction G β
      a✝ : α
      g : G
      yh : Membership.mem (MulAction.orbitRel.Quotient.orbit (Quotient.mk'' a✝)) ((f …
      h : G
      zh : Membership.mem (MulAction.orbitRel.Quotient.orbit (Quotient.mk'' a✝)) ((f …
      ⊢ Eq (HSMul.hSMul (HMul.hMul h (Inv.inv g)) ⟨(fun m => HSMul.hSMul m a✝) g, yh …
    -/
    ext
    /-
      case h.mk.mk.intro.intro.a
      G : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝² : Group G
      inst✝¹ : MulAction G α
      inst✝ : MulAction G β
      a✝ : α
      g : G
      yh : Membership.mem (MulAction.orbitRel.Quotient.orbit (Quotient.mk'' a✝)) ((f …
      h : G
      zh : Membership.mem (MulAction.orbitRel.Quotient.orbit (Quotient.mk'' a✝)) ((f …
      ⊢ Eq ↑(HSMul.hSMul (HMul.hMul h (Inv.inv g)) ⟨(fun m => HSMul.hSMul m a✝) g, y …
    -/
    simp [mul_smul]
    /-
      🎉 no goals
    -/


local notation "Ω" => orbitRel.Quotient G α


@[to_additive]
lemma _root_.Finite.of_finite_mulAction_orbitRel_quotient [Finite G] [Finite Ω] : Finite α := by
  /-
    G : Type u_1
    α : Type u_2
    inst✝³ : Group G
    inst✝² : MulAction G α
    inst✝¹ : Finite G
    inst✝ : Finite (MulAction.orbitRel.Quotient G α)
    ⊢ Finite α
  -/
  rw [(selfEquivSigmaOrbits' G _).finite_iff]
  have : ∀ g : Ω, Finite g.orbit := by
    intro g
    induction g using Quotient.inductionOn'
    simpa [Set.finite_coe_iff] using Finite.finite_mulAction_orbit _
  /-
    G : Type u_1
    α : Type u_2
    inst✝³ : Group G
    inst✝² : MulAction G α
    inst✝¹ : Finite G
    inst✝ : Finite (MulAction.orbitRel.Quotient G α)
    this : ∀ (g : MulAction.orbitRel.Quotient G α), Finite ↑g.orbit
    ⊢ Finite (Sigma fun ω => ↑ω.orbit)
  -/
  exact Finite.instSigma
  /-
    🎉 no goals
  -/


@[to_additive]
lemma orbitRel_le_fst :
    orbitRel G (α × β) ≤ (orbitRel G α).comap Prod.fst :=
  Setoid.le_def.2 fst_mem_orbit_of_mem_orbit


@[to_additive]
lemma orbitRel_le_snd :
    orbitRel G (α × β) ≤ (orbitRel G β).comap Prod.snd :=
  Setoid.le_def.2 snd_mem_orbit_of_mem_orbit


/-- If the stabilizer of `a` is `S`, then the stabilizer of `g • a` is `gSg⁻¹`. -/
theorem stabilizer_smul_eq_stabilizer_map_conj (g : G) (a : α) :
    stabilizer G (g • a) = (stabilizer G a).map (MulAut.conj g).toMonoidHom := by
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    g : G
    a : α
    ⊢ Eq (MulAction.stabilizer G (HSMul.hSMul g a)) (Subgroup.map (MulEquiv.toMono …
  -/
  ext h
  rw [mem_stabilizer_iff, ← smul_left_cancel_iff g⁻¹, smul_smul, smul_smul, smul_smul,
    inv_mul_cancel, one_smul, ← mem_stabilizer_iff, Subgroup.mem_map_equiv, MulAut.conj_symm_apply]


/-- A bijection between the stabilizers of two elements in the same orbit. -/
noncomputable def stabilizerEquivStabilizerOfOrbitRel {a b : α} (h : orbitRel G α a b) :
    stabilizer G a ≃* stabilizer G b :=
  let g : G := Classical.choose h
  have hg : g • b = a := Classical.choose_spec h
  have this : stabilizer G a = (stabilizer G b).map (MulAut.conj g).toMonoidHom := by
    /-
      G : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝² : Group G
      inst✝¹ : MulAction G α
      inst✝ : MulAction G β
      a b : α
      h : (MulAction.orbitRel G α) a b
      g : G := Classical.choose h
      hg : Eq (HSMul.hSMul g b) a
      ⊢ Eq (MulAction.stabilizer G a) (Subgroup.map (MulEquiv.toMonoidHom (MulAut.co …
    -/
    rw [← hg, stabilizer_smul_eq_stabilizer_map_conj]
    /-
      🎉 no goals
    -/
  (MulEquiv.subgroupCongr this).trans ((MulAut.conj g).subgroupMap <| stabilizer G b).symm


/-- If the stabilizer of `x` is `S`, then the stabilizer of `g +ᵥ x` is `g + S + (-g)`. -/
theorem stabilizer_vadd_eq_stabilizer_map_conj (g : G) (a : α) :
    stabilizer G (g +ᵥ a) = (stabilizer G a).map (AddAut.conj g).toAddMonoidHom := by
  /-
    G : Type u_1
    α : Type u_2
    inst✝¹ : AddGroup G
    inst✝ : AddAction G α
    g : G
    a : α
    ⊢ Eq (AddAction.stabilizer G (HVAdd.hVAdd g a)) (AddSubgroup.map (AddEquiv.toA …
  -/
  ext h
  rw [mem_stabilizer_iff, ← vadd_left_cancel_iff (-g), vadd_vadd, vadd_vadd, vadd_vadd,
    neg_add_cancel, zero_vadd, ← mem_stabilizer_iff, AddSubgroup.mem_map_equiv,
    AddAut.conj_symm_apply]


/-- A bijection between the stabilizers of two elements in the same orbit. -/
noncomputable def stabilizerEquivStabilizerOfOrbitRel {a b : α} (h : orbitRel G α a b) :
    stabilizer G a ≃+ stabilizer G b :=
  let g : G := Classical.choose h
  have hg : g +ᵥ b = a := Classical.choose_spec h
  have this : stabilizer G a = (stabilizer G b).map (AddAut.conj g).toAddMonoidHom := by
    /-
      G : Type u_1
      α : Type u_2
      inst✝¹ : AddGroup G
      inst✝ : AddAction G α
      a b : α
      h : (AddAction.orbitRel G α) a b
      g : G := Classical.choose h
      hg : Eq (HVAdd.hVAdd g b) a
      ⊢ Eq (AddAction.stabilizer G a) (AddSubgroup.map (AddEquiv.toAddMonoidHom (Add …
    -/
    rw [← hg, stabilizer_vadd_eq_stabilizer_map_conj]
    /-
      🎉 no goals
    -/
  (AddEquiv.addSubgroupCongr this).trans ((AddAut.conj g).addSubgroupMap <| stabilizer G b).symm


attribute [to_additive existing] MulAction.stabilizerEquivStabilizerOfOrbitRel


theorem Equiv.swap_mem_stabilizer {α : Type*} [DecidableEq α] {S : Set α} {a b : α} :
    Equiv.swap a b ∈ MulAction.stabilizer (Equiv.Perm α) S ↔ (a ∈ S ↔ b ∈ S) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    S : Set α
    a b : α
    ⊢ Iff (Membership.mem (MulAction.stabilizer (Equiv.Perm α) S) (Equiv.swap a b) …
  -/
  rw [MulAction.mem_stabilizer_iff, Set.ext_iff, ← swap_inv]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    S : Set α
    a b : α
    ⊢ Iff (∀ (x : α), Iff (Membership.mem (HSMul.hSMul (Inv.inv (Equiv.swap a b))  …
  -/
  simp_rw [Set.mem_inv_smul_set_iff, Perm.smul_def, swap_apply_def]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    S : Set α
    a b : α
    ⊢ Iff (∀ (x : α), Iff (Membership.mem S (ite (Eq x a) b (ite (Eq x b) a x))) ( …
  -/
  exact ⟨fun h ↦ by simpa [Iff.comm] using h a, by intros; split_ifs <;> simp [*]⟩
  /-
    🎉 no goals
  -/



/-- To prove inclusion of a *subgroup* in a stabilizer, it is enough to prove inclusions.-/
theorem le_stabilizer_iff_smul_le (s : Set α) (H : Subgroup G) :
    H ≤ stabilizer G s ↔ ∀ g ∈ H, g • s ⊆ s := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    α : Type u_2
    inst✝ : MulAction G α
    s : Set α
    H : Subgroup G
    ⊢ Iff (LE.le H (MulAction.stabilizer G s)) (∀ (g : G), Membership.mem H g → Ha …
  -/
  constructor
    /-
      case mp
      G : Type u_1
      inst✝¹ : Group G
      α : Type u_2
      inst✝ : MulAction G α
      s : Set α
      H : Subgroup G
      ⊢ LE.le H (MulAction.stabilizer G s) → ∀ (g : G), Membership.mem H g → HasSubs …
    -/
  · intro hyp g hg
    /-
      case mp
      G : Type u_1
      inst✝¹ : Group G
      α : Type u_2
      inst✝ : MulAction G α
      s : Set α
      H : Subgroup G
      hyp : LE.le H (MulAction.stabilizer G s)
      g : G
      hg : Membership.mem H g
      ⊢ HasSubset.Subset (HSMul.hSMul g s) s
    -/
    apply Eq.subset
    /-
      case mp.a
      G : Type u_1
      inst✝¹ : Group G
      α : Type u_2
      inst✝ : MulAction G α
      s : Set α
      H : Subgroup G
      hyp : LE.le H (MulAction.stabilizer G s)
      g : G
      hg : Membership.mem H g
      ⊢ Eq (HSMul.hSMul g s) s
    -/
    rw [← mem_stabilizer_iff]
    /-
      case mp.a
      G : Type u_1
      inst✝¹ : Group G
      α : Type u_2
      inst✝ : MulAction G α
      s : Set α
      H : Subgroup G
      hyp : LE.le H (MulAction.stabilizer G s)
      g : G
      hg : Membership.mem H g
      ⊢ Membership.mem (MulAction.stabilizer G s) g
    -/
    exact hyp hg
    /-
      🎉 no goals
    -/
    /-
      case mpr
      G : Type u_1
      inst✝¹ : Group G
      α : Type u_2
      inst✝ : MulAction G α
      s : Set α
      H : Subgroup G
      ⊢ (∀ (g : G), Membership.mem H g → HasSubset.Subset (HSMul.hSMul g s) s) → LE. …
    -/
  · intro hyp g hg
    /-
      case mpr
      G : Type u_1
      inst✝¹ : Group G
      α : Type u_2
      inst✝ : MulAction G α
      s : Set α
      H : Subgroup G
      hyp : ∀ (g : G), Membership.mem H g → HasSubset.Subset (HSMul.hSMul g s) s
      g : G
      hg : Membership.mem H g
      ⊢ Membership.mem (MulAction.stabilizer G s) g
    -/
    rw [mem_stabilizer_iff]
    /-
      case mpr
      G : Type u_1
      inst✝¹ : Group G
      α : Type u_2
      inst✝ : MulAction G α
      s : Set α
      H : Subgroup G
      hyp : ∀ (g : G), Membership.mem H g → HasSubset.Subset (HSMul.hSMul g s) s
      g : G
      hg : Membership.mem H g
      ⊢ Eq (HSMul.hSMul g s) s
    -/
    apply subset_antisymm (hyp g hg)
    /-
      case mpr
      G : Type u_1
      inst✝¹ : Group G
      α : Type u_2
      inst✝ : MulAction G α
      s : Set α
      H : Subgroup G
      hyp : ∀ (g : G), Membership.mem H g → HasSubset.Subset (HSMul.hSMul g s) s
      g : G
      hg : Membership.mem H g
      ⊢ HasSubset.Subset s (HSMul.hSMul g s)
    -/
    intro x hx
    /-
      case mpr
      G : Type u_1
      inst✝¹ : Group G
      α : Type u_2
      inst✝ : MulAction G α
      s : Set α
      H : Subgroup G
      hyp : ∀ (g : G), Membership.mem H g → HasSubset.Subset (HSMul.hSMul g s) s
      g : G
      hg : Membership.mem H g
      x : α
      hx : Membership.mem s x
      ⊢ Membership.mem (HSMul.hSMul g s) x
    -/
    use g⁻¹ • x
    /-
      case h
      G : Type u_1
      inst✝¹ : Group G
      α : Type u_2
      inst✝ : MulAction G α
      s : Set α
      H : Subgroup G
      hyp : ∀ (g : G), Membership.mem H g → HasSubset.Subset (HSMul.hSMul g s) s
      g : G
      hg : Membership.mem H g
      x : α
      hx : Membership.mem s x
      ⊢ And (Membership.mem s (HSMul.hSMul (Inv.inv g) x)) (Eq ((fun x => HSMul.hSMu …
    -/
    constructor
      /-
        case h.left
        G : Type u_1
        inst✝¹ : Group G
        α : Type u_2
        inst✝ : MulAction G α
        s : Set α
        H : Subgroup G
        hyp : ∀ (g : G), Membership.mem H g → HasSubset.Subset (HSMul.hSMul g s) s
        g : G
        hg : Membership.mem H g
        x : α
        hx : Membership.mem s x
        ⊢ Membership.mem s (HSMul.hSMul (Inv.inv g) x)
      -/
    · apply hyp g⁻¹ (inv_mem hg)
      /-
        case h.left.a
        G : Type u_1
        inst✝¹ : Group G
        α : Type u_2
        inst✝ : MulAction G α
        s : Set α
        H : Subgroup G
        hyp : ∀ (g : G), Membership.mem H g → HasSubset.Subset (HSMul.hSMul g s) s
        g : G
        hg : Membership.mem H g
        x : α
        hx : Membership.mem s x
        ⊢ Membership.mem (HSMul.hSMul (Inv.inv g) s) (HSMul.hSMul (Inv.inv g) x)
      -/
      simp only [Set.smul_mem_smul_set_iff, hx]
      /-
        🎉 no goals
      -/
      /-
        case h.right
        G : Type u_1
        inst✝¹ : Group G
        α : Type u_2
        inst✝ : MulAction G α
        s : Set α
        H : Subgroup G
        hyp : ∀ (g : G), Membership.mem H g → HasSubset.Subset (HSMul.hSMul g s) s
        g : G
        hg : Membership.mem H g
        x : α
        hx : Membership.mem s x
        ⊢ Eq ((fun x => HSMul.hSMul g x) (HSMul.hSMul (Inv.inv g) x)) x
      -/
    · simp only [smul_inv_smul]
      /-
        🎉 no goals
      -/


variable {M} in
lemma Module.stabilizer_units_eq_bot_of_ne_zero {x : M} (hx : x ≠ 0) :
    MulAction.stabilizer Rˣ x = ⊥ := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    x : M
    hx : Ne x 0
    ⊢ Eq (MulAction.stabilizer (Units R) x) Bot.bot
  -/
  rw [eq_bot_iff]
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    x : M
    hx : Ne x 0
    ⊢ LE.le (MulAction.stabilizer (Units R) x) Bot.bot
  -/
  intro g (hg : g.val • x = x)
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    x : M
    hx : Ne x 0
    g : Units R
    hg : Eq (HSMul.hSMul (↑g) x) x
    ⊢ Membership.mem Bot.bot g
  -/
  ext
  /-
    case a
    R : Type u_1
    M : Type u_2
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    x : M
    hx : Ne x 0
    g : Units R
    hg : Eq (HSMul.hSMul (↑g) x) x
    ⊢ Eq ↑g ↑1
  -/
  rw [← sub_eq_zero, ← smul_eq_zero_iff_left hx, Units.val_one, sub_smul, hg, one_smul, sub_self]
  /-
    🎉 no goals
  -/


