/-- The Frattini subgroup of a group is the intersection of the maximal subgroups. -/
def frattini (G : Type*) [Group G] : Subgroup G :=
  Order.radical (Subgroup G)


lemma frattini_le_coatom {K : Subgroup G} (h : IsCoatom K) : frattini G ≤ K :=
  Order.radical_le_coatom h


lemma frattini_le_comap_frattini_of_surjective (hφ : Function.Surjective φ) :
    frattini G ≤ (frattini H).comap φ := by
  /-
    G : Type u_1
    H : Type u_2
    inst✝¹ : Group G
    inst✝ : Group H
    φ : MonoidHom G H
    hφ : Function.Surjective ⇑φ
    ⊢ LE.le (frattini G) (Subgroup.comap φ (frattini H))
  -/
  simp_rw [frattini, Order.radical, comap_iInf, le_iInf_iff]
  /-
    G : Type u_1
    H : Type u_2
    inst✝¹ : Group G
    inst✝ : Group H
    φ : MonoidHom G H
    hφ : Function.Surjective ⇑φ
    ⊢ ∀ (i : Subgroup H), Membership.mem (setOf fun H_1 => IsCoatom H_1) i → LE.le …
  -/
  intro M hM
  /-
    G : Type u_1
    H : Type u_2
    inst✝¹ : Group G
    inst✝ : Group H
    φ : MonoidHom G H
    hφ : Function.Surjective ⇑φ
    M : Subgroup H
    hM : Membership.mem (setOf fun H_1 => IsCoatom H_1) M
    ⊢ LE.le (iInf fun a => iInf fun h => a) (Subgroup.comap φ M)
  -/
  apply biInf_le
  /-
    case hi
    G : Type u_1
    H : Type u_2
    inst✝¹ : Group G
    inst✝ : Group H
    φ : MonoidHom G H
    hφ : Function.Surjective ⇑φ
    M : Subgroup H
    hM : Membership.mem (setOf fun H_1 => IsCoatom H_1) M
    ⊢ Membership.mem (setOf fun H => IsCoatom H) (Subgroup.comap φ M)
  -/
  exact isCoatom_comap_of_surjective hφ hM
  /-
    🎉 no goals
  -/


/-- The Frattini subgroup is characteristic. -/
instance frattini_characteristic : (frattini G).Characteristic := by
  /-
    G : Type u_1
    H : Type u_2
    inst✝¹ : Group G
    inst✝ : Group H
    φ : MonoidHom G H
    ⊢ (frattini G).Characteristic
  -/
  rw [characteristic_iff_comap_eq]
  /-
    G : Type u_1
    H : Type u_2
    inst✝¹ : Group G
    inst✝ : Group H
    φ : MonoidHom G H
    ⊢ ∀ (ϕ : MulEquiv G G), Eq (Subgroup.comap ϕ.toMonoidHom (frattini G)) (fratti …
  -/
  intro φ
  /-
    G : Type u_1
    H : Type u_2
    inst✝¹ : Group G
    inst✝ : Group H
    φ✝ : MonoidHom G H
    φ : MulEquiv G G
    ⊢ Eq (Subgroup.comap φ.toMonoidHom (frattini G)) (frattini G)
  -/
  apply φ.comapSubgroup.map_radical
  /-
    🎉 no goals
  -/


/--
The Frattini subgroup consists of "non-generating" elements in the following sense:

If a subgroup together with the Frattini subgroup generates the whole group,
then the subgroup is already the whole group.
-/
theorem frattini_nongenerating [IsCoatomic (Subgroup G)] {K : Subgroup G}
    (h : K ⊔ frattini G = ⊤) : K = ⊤ :=
  Order.radical_nongenerating h


/-- When `G` is finite, the Frattini subgroup is nilpotent. -/
theorem frattini_nilpotent [Finite G] : Group.IsNilpotent (frattini G) := by
  -- We use the characterisation of nilpotency in terms of all Sylow subgroups being normal.
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite G
    ⊢ Group.IsNilpotent (Subtype fun x => Membership.mem (frattini G) x)
  -/
  have q := (isNilpotent_of_finite_tfae (G := frattini G)).out 0 3
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite G
    q : Iff (Group.IsNilpotent (Subtype fun x => Membership.mem (frattini G) x)) ( …
    ⊢ Group.IsNilpotent (Subtype fun x => Membership.mem (frattini G) x)
  -/
  rw [q]; clear q
  -- Consider each prime `p` and Sylow `p`-subgroup `P` of `frattini G`.
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite G
    ⊢ ∀ (p : Nat), Fact (Nat.Prime p) → ∀ (P : Sylow p (Subtype fun x => Membershi …
  -/
  intro p p_prime P
  -- The Frattini argument shows that the normalizer of `P` in `G`
  -- together with `frattini G` generates `G`.
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite G
    p : Nat
    p_prime : Fact (Nat.Prime p)
    P : Sylow p (Subtype fun x => Membership.mem (frattini G) x)
    ⊢ (↑P).Normal
  -/
  have frattini_argument := Sylow.normalizer_sup_eq_top P
  -- and hence by the nongenerating property of the Frattini subgroup that
  -- the normalizer of `P` in `G` is `G`.
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite G
    p : Nat
    p_prime : Fact (Nat.Prime p)
    P : Sylow p (Subtype fun x => Membership.mem (frattini G) x)
    frattini_argument : Eq (Max.max (Subgroup.map (frattini G).subtype ↑P).normali …
    ⊢ (↑P).Normal
  -/
  have normalizer_P := frattini_nongenerating frattini_argument
  -- This means that `P` is normal as a subgroup of `G`
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite G
    p : Nat
    p_prime : Fact (Nat.Prime p)
    P : Sylow p (Subtype fun x => Membership.mem (frattini G) x)
    frattini_argument : Eq (Max.max (Subgroup.map (frattini G).subtype ↑P).normali …
    normalizer_P : Eq (Subgroup.map (frattini G).subtype ↑P).normalizer Top.top
    ⊢ (↑P).Normal
  -/
  have P_normal_in_G : (map (frattini G).subtype P).Normal := normalizer_eq_top_iff.mp normalizer_P
  -- and hence also as a subgroup of `frattini G`, which was the remaining goal.
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite G
    p : Nat
    p_prime : Fact (Nat.Prime p)
    P : Sylow p (Subtype fun x => Membership.mem (frattini G) x)
    frattini_argument : Eq (Max.max (Subgroup.map (frattini G).subtype ↑P).normali …
    normalizer_P : Eq (Subgroup.map (frattini G).subtype ↑P).normalizer Top.top
    P_normal_in_G : (Subgroup.map (frattini G).subtype ↑P).Normal
    ⊢ (↑P).Normal
  -/
  exact P_normal_in_G.of_map_subtype
  /-
    🎉 no goals
  -/

