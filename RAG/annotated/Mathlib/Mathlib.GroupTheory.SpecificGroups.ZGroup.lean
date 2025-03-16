/-- A Z-group is a group whose Sylow subgroups are all cyclic. -/
@[mk_iff] class IsZGroup : Prop where
  isZGroup : ∀ p : ℕ, p.Prime → ∀ P : Sylow p G, IsCyclic P


instance [IsZGroup G] {p : ℕ} [Fact p.Prime] (P : Sylow p G) : IsCyclic P :=
  isZGroup p Fact.out P


theorem _root_.IsPGroup.isCyclic_of_isZGroup [IsZGroup G] {p : ℕ} [Fact p.Prime]
    {P : Subgroup G} (hP : IsPGroup p P) : IsCyclic P := by
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : IsZGroup G
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    P : Subgroup G
    hP : IsPGroup p (Subtype fun x => Membership.mem P x)
    ⊢ IsCyclic (Subtype fun x => Membership.mem P x)
  -/
  obtain ⟨Q, hQ⟩ := hP.exists_le_sylow
  /-
    case intro
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : IsZGroup G
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    P : Subgroup G
    hP : IsPGroup p (Subtype fun x => Membership.mem P x)
    Q : Sylow p G
    hQ : LE.le P ↑Q
    ⊢ IsCyclic (Subtype fun x => Membership.mem P x)
  -/
  exact Subgroup.isCyclic_of_le hQ
  /-
    🎉 no goals
  -/


theorem of_squarefree (hG : Squarefree (Nat.card G)) : IsZGroup G := by
  /-
    G : Type u_1
    inst✝ : Group G
    hG : Squarefree (Nat.card G)
    ⊢ IsZGroup G
  -/
  have : Finite G := Nat.finite_of_card_ne_zero hG.ne_zero
  /-
    G : Type u_1
    inst✝ : Group G
    hG : Squarefree (Nat.card G)
    this : Finite G
    ⊢ IsZGroup G
  -/
  refine ⟨fun p hp P ↦ ?_⟩
  /-
    G : Type u_1
    inst✝ : Group G
    hG : Squarefree (Nat.card G)
    this : Finite G
    p : Nat
    hp : Nat.Prime p
    P : Sylow p G
    ⊢ IsCyclic (Subtype fun x => Membership.mem (↑P) x)
  -/
  have := Fact.mk hp
  /-
    G : Type u_1
    inst✝ : Group G
    hG : Squarefree (Nat.card G)
    this✝ : Finite G
    p : Nat
    hp : Nat.Prime p
    P : Sylow p G
    this : Fact (Nat.Prime p)
    ⊢ IsCyclic (Subtype fun x => Membership.mem (↑P) x)
  -/
  obtain ⟨k, hk⟩ := P.2.exists_card_eq
  /-
    case intro
    G : Type u_1
    inst✝ : Group G
    hG : Squarefree (Nat.card G)
    this✝ : Finite G
    p : Nat
    hp : Nat.Prime p
    P : Sylow p G
    this : Fact (Nat.Prime p)
    k : Nat
    hk : Eq (Nat.card (Subtype fun x => Membership.mem (↑P) x)) (HPow.hPow p k)
    ⊢ IsCyclic (Subtype fun x => Membership.mem (↑P) x)
  -/
  exact isCyclic_of_card_dvd_prime ((hk ▸ hG.pow_dvd_of_pow_dvd) P.card_subgroup_dvd_card)
  /-
    🎉 no goals
  -/


theorem of_injective [hG' : IsZGroup G'] (hf : Function.Injective f) : IsZGroup G := by
  /-
    G : Type u_1
    G' : Type u_2
    inst✝¹ : Group G
    inst✝ : Group G'
    f : MonoidHom G G'
    hG' : IsZGroup G'
    hf : Function.Injective ⇑f
    ⊢ IsZGroup G
  -/
  rw [isZGroup_iff] at hG' ⊢
  /-
    G : Type u_1
    G' : Type u_2
    inst✝¹ : Group G
    inst✝ : Group G'
    f : MonoidHom G G'
    hG' : ∀ (p : Nat), Nat.Prime p → ∀ (P : Sylow p G'), IsCyclic (Subtype fun x = …
    hf : Function.Injective ⇑f
    ⊢ ∀ (p : Nat), Nat.Prime p → ∀ (P : Sylow p G), IsCyclic (Subtype fun x => Mem …
  -/
  intro p hp P
  /-
    G : Type u_1
    G' : Type u_2
    inst✝¹ : Group G
    inst✝ : Group G'
    f : MonoidHom G G'
    hG' : ∀ (p : Nat), Nat.Prime p → ∀ (P : Sylow p G'), IsCyclic (Subtype fun x = …
    hf : Function.Injective ⇑f
    p : Nat
    hp : Nat.Prime p
    P : Sylow p G
    ⊢ IsCyclic (Subtype fun x => Membership.mem (↑P) x)
  -/
  obtain ⟨Q, hQ⟩ := P.exists_comap_eq_of_injective hf
  /-
    case intro
    G : Type u_1
    G' : Type u_2
    inst✝¹ : Group G
    inst✝ : Group G'
    f : MonoidHom G G'
    hG' : ∀ (p : Nat), Nat.Prime p → ∀ (P : Sylow p G'), IsCyclic (Subtype fun x = …
    hf : Function.Injective ⇑f
    p : Nat
    hp : Nat.Prime p
    P : Sylow p G
    Q : Sylow p G'
    hQ : Eq (Subgroup.comap f ↑Q) ↑P
    ⊢ IsCyclic (Subtype fun x => Membership.mem (↑P) x)
  -/
  specialize hG' p hp Q
  /-
    case intro
    G : Type u_1
    G' : Type u_2
    inst✝¹ : Group G
    inst✝ : Group G'
    f : MonoidHom G G'
    hf : Function.Injective ⇑f
    p : Nat
    hp : Nat.Prime p
    P : Sylow p G
    Q : Sylow p G'
    hQ : Eq (Subgroup.comap f ↑Q) ↑P
    hG' : IsCyclic (Subtype fun x => Membership.mem (↑Q) x)
    ⊢ IsCyclic (Subtype fun x => Membership.mem (↑P) x)
  -/
  have h : Subgroup.map f P ≤ Q := hQ ▸ Subgroup.map_comap_le f ↑Q
  /-
    case intro
    G : Type u_1
    G' : Type u_2
    inst✝¹ : Group G
    inst✝ : Group G'
    f : MonoidHom G G'
    hf : Function.Injective ⇑f
    p : Nat
    hp : Nat.Prime p
    P : Sylow p G
    Q : Sylow p G'
    hQ : Eq (Subgroup.comap f ↑Q) ↑P
    hG' : IsCyclic (Subtype fun x => Membership.mem (↑Q) x)
    h : LE.le (Subgroup.map f ↑P) ↑Q
    ⊢ IsCyclic (Subtype fun x => Membership.mem (↑P) x)
  -/
  have := isCyclic_of_surjective _ (Subgroup.subgroupOfEquivOfLe h).surjective
  /-
    case intro
    G : Type u_1
    G' : Type u_2
    inst✝¹ : Group G
    inst✝ : Group G'
    f : MonoidHom G G'
    hf : Function.Injective ⇑f
    p : Nat
    hp : Nat.Prime p
    P : Sylow p G
    Q : Sylow p G'
    hQ : Eq (Subgroup.comap f ↑Q) ↑P
    hG' : IsCyclic (Subtype fun x => Membership.mem (↑Q) x)
    h : LE.le (Subgroup.map f ↑P) ↑Q
    this : IsCyclic (Subtype fun x => Membership.mem (Subgroup.map f ↑P) x)
    ⊢ IsCyclic (Subtype fun x => Membership.mem (↑P) x)
  -/
  exact isCyclic_of_surjective _ (Subgroup.equivMapOfInjective P f hf).symm.surjective
  /-
    🎉 no goals
  -/


instance [IsZGroup G] (H : Subgroup G) : IsZGroup H := of_injective H.subtype_injective


theorem of_surjective [Finite G] [hG : IsZGroup G] (hf : Function.Surjective f) : IsZGroup G' := by
  /-
    G : Type u_1
    G' : Type u_2
    inst✝² : Group G
    inst✝¹ : Group G'
    f : MonoidHom G G'
    inst✝ : Finite G
    hG : IsZGroup G
    hf : Function.Surjective ⇑f
    ⊢ IsZGroup G'
  -/
  rw [isZGroup_iff] at hG ⊢
  /-
    G : Type u_1
    G' : Type u_2
    inst✝² : Group G
    inst✝¹ : Group G'
    f : MonoidHom G G'
    inst✝ : Finite G
    hG : ∀ (p : Nat), Nat.Prime p → ∀ (P : Sylow p G), IsCyclic (Subtype fun x =>  …
    hf : Function.Surjective ⇑f
    ⊢ ∀ (p : Nat), Nat.Prime p → ∀ (P : Sylow p G'), IsCyclic (Subtype fun x => Me …
  -/
  intro p hp P
  /-
    G : Type u_1
    G' : Type u_2
    inst✝² : Group G
    inst✝¹ : Group G'
    f : MonoidHom G G'
    inst✝ : Finite G
    hG : ∀ (p : Nat), Nat.Prime p → ∀ (P : Sylow p G), IsCyclic (Subtype fun x =>  …
    hf : Function.Surjective ⇑f
    p : Nat
    hp : Nat.Prime p
    P : Sylow p G'
    ⊢ IsCyclic (Subtype fun x => Membership.mem (↑P) x)
  -/
  have := Fact.mk hp
  /-
    G : Type u_1
    G' : Type u_2
    inst✝² : Group G
    inst✝¹ : Group G'
    f : MonoidHom G G'
    inst✝ : Finite G
    hG : ∀ (p : Nat), Nat.Prime p → ∀ (P : Sylow p G), IsCyclic (Subtype fun x =>  …
    hf : Function.Surjective ⇑f
    p : Nat
    hp : Nat.Prime p
    P : Sylow p G'
    this : Fact (Nat.Prime p)
    ⊢ IsCyclic (Subtype fun x => Membership.mem (↑P) x)
  -/
  obtain ⟨Q, rfl⟩ := Sylow.mapSurjective_surjective hf p P
  /-
    case intro
    G : Type u_1
    G' : Type u_2
    inst✝² : Group G
    inst✝¹ : Group G'
    f : MonoidHom G G'
    inst✝ : Finite G
    hG : ∀ (p : Nat), Nat.Prime p → ∀ (P : Sylow p G), IsCyclic (Subtype fun x =>  …
    hf : Function.Surjective ⇑f
    p : Nat
    hp : Nat.Prime p
    this : Fact (Nat.Prime p)
    Q : Sylow p G
    ⊢ IsCyclic (Subtype fun x => Membership.mem (↑(Sylow.mapSurjective hf Q)) x)
  -/
  specialize hG p hp Q
  /-
    case intro
    G : Type u_1
    G' : Type u_2
    inst✝² : Group G
    inst✝¹ : Group G'
    f : MonoidHom G G'
    inst✝ : Finite G
    hf : Function.Surjective ⇑f
    p : Nat
    hp : Nat.Prime p
    this : Fact (Nat.Prime p)
    Q : Sylow p G
    hG : IsCyclic (Subtype fun x => Membership.mem (↑Q) x)
    ⊢ IsCyclic (Subtype fun x => Membership.mem (↑(Sylow.mapSurjective hf Q)) x)
  -/
  exact isCyclic_of_surjective _ (f.subgroupMap_surjective Q)
  /-
    🎉 no goals
  -/


instance [Finite G] [IsZGroup G] (H : Subgroup G) [H.Normal] : IsZGroup (G ⧸ H) :=
  of_surjective (QuotientGroup.mk'_surjective H)


variable (G) in
theorem commutator_lt [Finite G] [IsZGroup G] [Nontrivial G] : commutator G < ⊤ := by
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : Finite G
    inst✝¹ : IsZGroup G
    inst✝ : Nontrivial G
    ⊢ LT.lt (commutator G) Top.top
  -/
  let p := (Nat.card G).minFac
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : Finite G
    inst✝¹ : IsZGroup G
    inst✝ : Nontrivial G
    p : Nat := (Nat.card G).minFac
    ⊢ LT.lt (commutator G) Top.top
  -/
  have hp : p.Prime := Nat.minFac_prime Finite.one_lt_card.ne'
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : Finite G
    inst✝¹ : IsZGroup G
    inst✝ : Nontrivial G
    p : Nat := (Nat.card G).minFac
    hp : Nat.Prime p
    ⊢ LT.lt (commutator G) Top.top
  -/
  have := Fact.mk hp
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : Finite G
    inst✝¹ : IsZGroup G
    inst✝ : Nontrivial G
    p : Nat := (Nat.card G).minFac
    hp : Nat.Prime p
    this : Fact (Nat.Prime p)
    ⊢ LT.lt (commutator G) Top.top
  -/
  let P : Sylow p G := default
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : Finite G
    inst✝¹ : IsZGroup G
    inst✝ : Nontrivial G
    p : Nat := (Nat.card G).minFac
    hp : Nat.Prime p
    this : Fact (Nat.Prime p)
    P : Sylow p G := Inhabited.default
    ⊢ LT.lt (commutator G) Top.top
  -/
  have hP := isZGroup p hp P
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : Finite G
    inst✝¹ : IsZGroup G
    inst✝ : Nontrivial G
    p : Nat := (Nat.card G).minFac
    hp : Nat.Prime p
    this : Fact (Nat.Prime p)
    P : Sylow p G := Inhabited.default
    hP : IsCyclic (Subtype fun x => Membership.mem (↑P) x)
    ⊢ LT.lt (commutator G) Top.top
  -/
  let f := MonoidHom.transferSylow P (hP.normalizer_le_centralizer rfl)
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : Finite G
    inst✝¹ : IsZGroup G
    inst✝ : Nontrivial G
    p : Nat := (Nat.card G).minFac
    hp : Nat.Prime p
    this : Fact (Nat.Prime p)
    P : Sylow p G := Inhabited.default
    hP : IsCyclic (Subtype fun x => Membership.mem (↑P) x)
    f : MonoidHom G (Subtype fun x => Membership.mem (↑P) x) := MonoidHom.transfer …
    ⊢ LT.lt (commutator G) Top.top
  -/
  refine lt_of_le_of_lt (Abelianization.commutator_subset_ker f) ?_
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : Finite G
    inst✝¹ : IsZGroup G
    inst✝ : Nontrivial G
    p : Nat := (Nat.card G).minFac
    hp : Nat.Prime p
    this : Fact (Nat.Prime p)
    P : Sylow p G := Inhabited.default
    hP : IsCyclic (Subtype fun x => Membership.mem (↑P) x)
    f : MonoidHom G (Subtype fun x => Membership.mem (↑P) x) := MonoidHom.transfer …
    ⊢ LT.lt f.ker Top.top
  -/
  have h := P.ne_bot_of_dvd_card (Nat.card G).minFac_dvd
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : Finite G
    inst✝¹ : IsZGroup G
    inst✝ : Nontrivial G
    p : Nat := (Nat.card G).minFac
    hp : Nat.Prime p
    this : Fact (Nat.Prime p)
    P : Sylow p G := Inhabited.default
    hP : IsCyclic (Subtype fun x => Membership.mem (↑P) x)
    f : MonoidHom G (Subtype fun x => Membership.mem (↑P) x) := MonoidHom.transfer …
    h : Ne (↑P) Bot.bot
    ⊢ LT.lt f.ker Top.top
  -/
  contrapose! h
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : Finite G
    inst✝¹ : IsZGroup G
    inst✝ : Nontrivial G
    p : Nat := (Nat.card G).minFac
    hp : Nat.Prime p
    this : Fact (Nat.Prime p)
    P : Sylow p G := Inhabited.default
    hP : IsCyclic (Subtype fun x => Membership.mem (↑P) x)
    f : MonoidHom G (Subtype fun x => Membership.mem (↑P) x) := MonoidHom.transfer …
    h : Not (LT.lt f.ker Top.top)
    ⊢ Eq (↑P) Bot.bot
  -/
  rw [← Subgroup.isComplement'_top_left, ← (not_lt_top_iff.mp h)]
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : Finite G
    inst✝¹ : IsZGroup G
    inst✝ : Nontrivial G
    p : Nat := (Nat.card G).minFac
    hp : Nat.Prime p
    this : Fact (Nat.Prime p)
    P : Sylow p G := Inhabited.default
    hP : IsCyclic (Subtype fun x => Membership.mem (↑P) x)
    f : MonoidHom G (Subtype fun x => Membership.mem (↑P) x) := MonoidHom.transfer …
    h : Not (LT.lt f.ker Top.top)
    ⊢ f.ker.IsComplement' ↑P
  -/
  exact hP.isComplement' rfl
  /-
    🎉 no goals
  -/


instance [Finite G] [IsZGroup G] : IsSolvable G := by
  /-
    G : Type u_1
    G' : Type u_2
    G'' : Type u_3
    inst✝⁴ : Group G
    inst✝³ : Group G'
    inst✝² : Group G''
    f : MonoidHom G G'
    f' : MonoidHom G' G''
    inst✝¹ : Finite G
    inst✝ : IsZGroup G
    ⊢ IsSolvable G
  -/
  rw [isSolvable_iff_commutator_lt]
  /-
    G : Type u_1
    G' : Type u_2
    G'' : Type u_3
    inst✝⁴ : Group G
    inst✝³ : Group G'
    inst✝² : Group G''
    f : MonoidHom G G'
    f' : MonoidHom G' G''
    inst✝¹ : Finite G
    inst✝ : IsZGroup G
    ⊢ ∀ (H : Subgroup G), Ne H Bot.bot → LT.lt (Bracket.bracket H H) H
  -/
  intro H h
  /-
    G : Type u_1
    G' : Type u_2
    G'' : Type u_3
    inst✝⁴ : Group G
    inst✝³ : Group G'
    inst✝² : Group G''
    f : MonoidHom G G'
    f' : MonoidHom G' G''
    inst✝¹ : Finite G
    inst✝ : IsZGroup G
    H : Subgroup G
    h : Ne H Bot.bot
    ⊢ LT.lt (Bracket.bracket H H) H
  -/
  rw [← H.nontrivial_iff_ne_bot] at h
  rw [← H.range_subtype, MonoidHom.range_eq_map, ← Subgroup.map_commutator,
    Subgroup.map_subtype_lt_map_subtype]
  /-
    G : Type u_1
    G' : Type u_2
    G'' : Type u_3
    inst✝⁴ : Group G
    inst✝³ : Group G'
    inst✝² : Group G''
    f : MonoidHom G G'
    f' : MonoidHom G' G''
    inst✝¹ : Finite G
    inst✝ : IsZGroup G
    H : Subgroup G
    h : Nontrivial (Subtype fun x => Membership.mem H x)
    ⊢ LT.lt (Bracket.bracket Top.top Top.top) Top.top
  -/
  exact commutator_lt H
  /-
    🎉 no goals
  -/


variable (G) in
theorem exponent_eq_card [Finite G] [IsZGroup G] : Monoid.exponent G = Nat.card G := by
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Finite G
    inst✝ : IsZGroup G
    ⊢ Eq (Monoid.exponent G) (Nat.card G)
  -/
  refine dvd_antisymm Group.exponent_dvd_nat_card ?_
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Finite G
    inst✝ : IsZGroup G
    ⊢ Dvd.dvd (Nat.card G) (Monoid.exponent G)
  -/
  rw [← Nat.factorization_prime_le_iff_dvd Nat.card_pos.ne' Monoid.exponent_ne_zero_of_finite]
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Finite G
    inst✝ : IsZGroup G
    ⊢ ∀ (p : Nat), Nat.Prime p → LE.le ((Nat.card G).factorization p) ((Monoid.exp …
  -/
  intro p hp
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Finite G
    inst✝ : IsZGroup G
    p : Nat
    hp : Nat.Prime p
    ⊢ LE.le ((Nat.card G).factorization p) ((Monoid.exponent G).factorization p)
  -/
  have := Fact.mk hp
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Finite G
    inst✝ : IsZGroup G
    p : Nat
    hp : Nat.Prime p
    this : Fact (Nat.Prime p)
    ⊢ LE.le ((Nat.card G).factorization p) ((Monoid.exponent G).factorization p)
  -/
  let P : Sylow p G := default
  rw [← hp.pow_dvd_iff_le_factorization Monoid.exponent_ne_zero_of_finite,
      ← P.card_eq_multiplicity, ← (isZGroup p hp P).exponent_eq_card]
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Finite G
    inst✝ : IsZGroup G
    p : Nat
    hp : Nat.Prime p
    this : Fact (Nat.Prime p)
    P : Sylow p G := Inhabited.default
    ⊢ Dvd.dvd (Monoid.exponent (Subtype fun x => Membership.mem (↑P) x)) (Monoid.e …
  -/
  exact Monoid.exponent_dvd_of_monoidHom P.1.subtype P.1.subtype_injective
  /-
    🎉 no goals
  -/


instance [Finite G] [IsZGroup G] [hG : Group.IsNilpotent G] : IsCyclic G := by
  have (p : { x // x ∈ (Nat.card G).primeFactors }) : Fact p.1.Prime :=
    ⟨Nat.prime_of_mem_primeFactors p.2⟩
  /-
    G : Type u_1
    G' : Type u_2
    G'' : Type u_3
    inst✝⁴ : Group G
    inst✝³ : Group G'
    inst✝² : Group G''
    f : MonoidHom G G'
    f' : MonoidHom G' G''
    inst✝¹ : Finite G
    inst✝ : IsZGroup G
    hG : Group.IsNilpotent G
    this : ∀ (p : Subtype fun x => Membership.mem (Nat.card G).primeFactors x), Fa …
    ⊢ IsCyclic G
  -/
  obtain ⟨ϕ⟩ := ((isNilpotent_of_finite_tfae (G := G)).out 0 4).mp hG
  let _ : CommGroup G :=
    ⟨fun g h ↦ by rw [← ϕ.symm.injective.eq_iff, map_mul, mul_comm, ← map_mul]⟩
  /-
    case intro
    G : Type u_1
    G' : Type u_2
    G'' : Type u_3
    inst✝⁴ : Group G
    inst✝³ : Group G'
    inst✝² : Group G''
    f : MonoidHom G G'
    f' : MonoidHom G' G''
    inst✝¹ : Finite G
    inst✝ : IsZGroup G
    hG : Group.IsNilpotent G
    this : ∀ (p : Subtype fun x => Membership.mem (Nat.card G).primeFactors x), Fa …
    ϕ : MulEquiv ((p : Subtype fun x => Membership.mem (Nat.card G).primeFactors x …
    x✝ : CommGroup G := CommGroup.mk ⋯
    ⊢ IsCyclic G
  -/
  exact IsCyclic.of_exponent_eq_card (exponent_eq_card G)
  /-
    🎉 no goals
  -/


/-- A finite Z-group has cyclic abelianization. -/
instance isCyclic_abelianization [Finite G] [IsZGroup G] : IsCyclic (Abelianization G) :=
  let _ : IsZGroup (Abelianization G) := inferInstanceAs (IsZGroup (G ⧸ commutator G))
  inferInstance


variable (G) in
/-- A finite Z-group has cyclic commutator subgroup. -/
theorem isCyclic_commutator [Finite G] [IsZGroup G] : IsCyclic (commutator G) := by
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Finite G
    inst✝ : IsZGroup G
    ⊢ IsCyclic (Subtype fun x => Membership.mem (commutator G) x)
  -/
  refine WellFoundedLT.induction (C := fun H ↦ IsCyclic (⁅H, H⁆ : Subgroup G)) (⊤ : Subgroup G) ?_
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Finite G
    inst✝ : IsZGroup G
    ⊢ ∀ (x : Subgroup G), (∀ (y : Subgroup G), LT.lt y x → (fun H => IsCyclic (Sub …
  -/
  intro H hH
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Finite G
    inst✝ : IsZGroup G
    H : Subgroup G
    hH : ∀ (y : Subgroup G), LT.lt y H → (fun H => IsCyclic (Subtype fun x => Memb …
    ⊢ IsCyclic (Subtype fun x => Membership.mem (Bracket.bracket H H) x)
  -/
  rcases eq_or_ne H ⊥ with rfl | h
    /-
      case inl
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : Finite G
      inst✝ : IsZGroup G
      hH : ∀ (y : Subgroup G), LT.lt y Bot.bot → (fun H => IsCyclic (Subtype fun x = …
      ⊢ IsCyclic (Subtype fun x => Membership.mem (Bracket.bracket Bot.bot Bot.bot) x)
    -/
  · rw [Subgroup.commutator_bot_left]
    /-
      case inl
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : Finite G
      inst✝ : IsZGroup G
      hH : ∀ (y : Subgroup G), LT.lt y Bot.bot → (fun H => IsCyclic (Subtype fun x = …
      ⊢ IsCyclic (Subtype fun x => Membership.mem Bot.bot x)
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case inr
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : Finite G
      inst✝ : IsZGroup G
      H : Subgroup G
      hH : ∀ (y : Subgroup G), LT.lt y H → (fun H => IsCyclic (Subtype fun x => Memb …
      h : Ne H Bot.bot
      ⊢ IsCyclic (Subtype fun x => Membership.mem (Bracket.bracket H H) x)
    -/
  · specialize hH ⁅H, H⁆ (IsSolvable.commutator_lt_of_ne_bot h)
    replace hH : IsCyclic (⁅commutator H, commutator H⁆ : Subgroup H) := by
      let f := Subgroup.equivMapOfInjective ⁅commutator H, commutator H⁆ _ H.subtype_injective
      rw [Subgroup.map_commutator, Subgroup.map_subtype_commutator] at f
      exact isCyclic_of_surjective f.symm f.symm.surjective
    suffices IsCyclic (commutator H) by
      let f := Subgroup.equivMapOfInjective (commutator H) _ H.subtype_injective
      rw [Subgroup.map_subtype_commutator] at f
      exact isCyclic_of_surjective f f.surjective
    suffices h : commutator (commutator H) ≤ Subgroup.center (commutator H) by
      rw [← Abelianization.ker_of (commutator H)] at h
      let _ := commGroupOfCyclicCenterQuotient Abelianization.of h
      infer_instance
    suffices h : (commutator (commutator H)).map (commutator H).subtype ≤
        Subgroup.centralizer (commutator H) by
      simpa [SetLike.le_def, Subgroup.mem_center_iff, Subgroup.mem_centralizer_iff] using h
    /-
      case inr
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : Finite G
      inst✝ : IsZGroup G
      H : Subgroup G
      h : Ne H Bot.bot
      hH : IsCyclic (Subtype fun x => Membership.mem (Bracket.bracket (commutator (S …
      ⊢ LE.le (Subgroup.map (commutator (Subtype fun x => Membership.mem H x)).subty …
    -/
    rw [Subgroup.map_subtype_commutator, Subgroup.le_centralizer_iff]
    /-
      case inr
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : Finite G
      inst✝ : IsZGroup G
      H : Subgroup G
      h : Ne H Bot.bot
      hH : IsCyclic (Subtype fun x => Membership.mem (Bracket.bracket (commutator (S …
      ⊢ LE.le (commutator (Subtype fun x => Membership.mem H x)) (Subgroup.centraliz …
    -/
    let _ := (hH.mulAutMulEquiv _).toMonoidHom.commGroupOfInjective (hH.mulAutMulEquiv _).injective
    /-
      case inr
      G : Type u_1
      inst✝² : Group G
      inst✝¹ : Finite G
      inst✝ : IsZGroup G
      H : Subgroup G
      h : Ne H Bot.bot
      hH : IsCyclic (Subtype fun x => Membership.mem (Bracket.bracket (commutator (S …
      x✝ : CommGroup (MulAut (Subtype fun x => Membership.mem (Bracket.bracket (comm …
      ⊢ LE.le (commutator (Subtype fun x => Membership.mem H x)) (Subgroup.centraliz …
    -/
    have h := Abelianization.commutator_subset_ker ⁅commutator H, commutator H⁆.normalizerMonoidHom
    rwa [Subgroup.normalizerMonoidHom_ker, Subgroup.normalizer_eq_top,
      ← Subgroup.map_subtype_le_map_subtype, Subgroup.map_subtype_commutator,
        Subgroup.map_subgroupOf_eq_of_le le_top] at h


/-- An extension of coprime Z-groups is a Z-group. -/
theorem isZGroup_of_coprime [Finite G] [IsZGroup G] [IsZGroup G'']
    (h_le : f'.ker ≤ f.range) (h_cop : (Nat.card G).Coprime (Nat.card G'')) :
    IsZGroup G' := by
  /-
    G : Type u_1
    G' : Type u_2
    G'' : Type u_3
    inst✝⁵ : Group G
    inst✝⁴ : Group G'
    inst✝³ : Group G''
    f : MonoidHom G G'
    f' : MonoidHom G' G''
    inst✝² : Finite G
    inst✝¹ : IsZGroup G
    inst✝ : IsZGroup G''
    h_le : LE.le f'.ker f.range
    h_cop : (Nat.card G).Coprime (Nat.card G'')
    ⊢ IsZGroup G'
  -/
  refine ⟨fun p hp P ↦ ?_⟩
  /-
    G : Type u_1
    G' : Type u_2
    G'' : Type u_3
    inst✝⁵ : Group G
    inst✝⁴ : Group G'
    inst✝³ : Group G''
    f : MonoidHom G G'
    f' : MonoidHom G' G''
    inst✝² : Finite G
    inst✝¹ : IsZGroup G
    inst✝ : IsZGroup G''
    h_le : LE.le f'.ker f.range
    h_cop : (Nat.card G).Coprime (Nat.card G'')
    p : Nat
    hp : Nat.Prime p
    P : Sylow p G'
    ⊢ IsCyclic (Subtype fun x => Membership.mem (↑P) x)
  -/
  have := Fact.mk hp
  replace h_cop := (h_cop.of_dvd ((Subgroup.card_dvd_of_le h_le).trans
    (Subgroup.card_range_dvd f)) (Subgroup.index_ker f' ▸ f'.range.card_subgroup_dvd_card))
  /-
    G : Type u_1
    G' : Type u_2
    G'' : Type u_3
    inst✝⁵ : Group G
    inst✝⁴ : Group G'
    inst✝³ : Group G''
    f : MonoidHom G G'
    f' : MonoidHom G' G''
    inst✝² : Finite G
    inst✝¹ : IsZGroup G
    inst✝ : IsZGroup G''
    h_le : LE.le f'.ker f.range
    p : Nat
    hp : Nat.Prime p
    P : Sylow p G'
    this : Fact (Nat.Prime p)
    h_cop : (Nat.card (Subtype fun x => Membership.mem f'.ker x)).Coprime f'.ker.i …
    ⊢ IsCyclic (Subtype fun x => Membership.mem (↑P) x)
  -/
  rcases P.2.le_or_disjoint_of_coprime h_cop with h | h
    /-
      case inl
      G : Type u_1
      G' : Type u_2
      G'' : Type u_3
      inst✝⁵ : Group G
      inst✝⁴ : Group G'
      inst✝³ : Group G''
      f : MonoidHom G G'
      f' : MonoidHom G' G''
      inst✝² : Finite G
      inst✝¹ : IsZGroup G
      inst✝ : IsZGroup G''
      h_le : LE.le f'.ker f.range
      p : Nat
      hp : Nat.Prime p
      P : Sylow p G'
      this : Fact (Nat.Prime p)
      h_cop : (Nat.card (Subtype fun x => Membership.mem f'.ker x)).Coprime f'.ker.i …
      h : LE.le (↑P) f'.ker
      ⊢ IsCyclic (Subtype fun x => Membership.mem (↑P) x)
    -/
  · replace h_le : P ≤ f.range := h.trans h_le
    suffices IsCyclic (P.subgroupOf f.range) by
      have key := Subgroup.subgroupOfEquivOfLe h_le
      exact isCyclic_of_surjective key key.surjective
    /-
      case inl
      G : Type u_1
      G' : Type u_2
      G'' : Type u_3
      inst✝⁵ : Group G
      inst✝⁴ : Group G'
      inst✝³ : Group G''
      f : MonoidHom G G'
      f' : MonoidHom G' G''
      inst✝² : Finite G
      inst✝¹ : IsZGroup G
      inst✝ : IsZGroup G''
      p : Nat
      hp : Nat.Prime p
      P : Sylow p G'
      this : Fact (Nat.Prime p)
      h_cop : (Nat.card (Subtype fun x => Membership.mem f'.ker x)).Coprime f'.ker.i …
      h : LE.le (↑P) f'.ker
      h_le : LE.le (↑P) f.range
      ⊢ IsCyclic (Subtype fun x => Membership.mem ((↑P).subgroupOf f.range) x)
    -/
    obtain ⟨Q, hQ⟩ := Sylow.mapSurjective_surjective f.rangeRestrict_surjective p (P.subtype h_le)
    /-
      case inl.intro
      G : Type u_1
      G' : Type u_2
      G'' : Type u_3
      inst✝⁵ : Group G
      inst✝⁴ : Group G'
      inst✝³ : Group G''
      f : MonoidHom G G'
      f' : MonoidHom G' G''
      inst✝² : Finite G
      inst✝¹ : IsZGroup G
      inst✝ : IsZGroup G''
      p : Nat
      hp : Nat.Prime p
      P : Sylow p G'
      this : Fact (Nat.Prime p)
      h_cop : (Nat.card (Subtype fun x => Membership.mem f'.ker x)).Coprime f'.ker.i …
      h : LE.le (↑P) f'.ker
      h_le : LE.le (↑P) f.range
      Q : Sylow p G
      hQ : Eq (Sylow.mapSurjective ⋯ Q) (P.subtype h_le)
      ⊢ IsCyclic (Subtype fun x => Membership.mem ((↑P).subgroupOf f.range) x)
    -/
    rw [Sylow.ext_iff, Sylow.coe_mapSurjective, Sylow.coe_subtype] at hQ
    /-
      case inl.intro
      G : Type u_1
      G' : Type u_2
      G'' : Type u_3
      inst✝⁵ : Group G
      inst✝⁴ : Group G'
      inst✝³ : Group G''
      f : MonoidHom G G'
      f' : MonoidHom G' G''
      inst✝² : Finite G
      inst✝¹ : IsZGroup G
      inst✝ : IsZGroup G''
      p : Nat
      hp : Nat.Prime p
      P : Sylow p G'
      this : Fact (Nat.Prime p)
      h_cop : (Nat.card (Subtype fun x => Membership.mem f'.ker x)).Coprime f'.ker.i …
      h : LE.le (↑P) f'.ker
      h_le : LE.le (↑P) f.range
      Q : Sylow p G
      hQ : Eq (Subgroup.map f.rangeRestrict ↑Q) ((↑P).subgroupOf f.range)
      ⊢ IsCyclic (Subtype fun x => Membership.mem ((↑P).subgroupOf f.range) x)
    -/
    exact hQ ▸ isCyclic_of_surjective _ (f.rangeRestrict.subgroupMap_surjective Q)
    /-
      🎉 no goals
    -/
    /-
      case inr
      G : Type u_1
      G' : Type u_2
      G'' : Type u_3
      inst✝⁵ : Group G
      inst✝⁴ : Group G'
      inst✝³ : Group G''
      f : MonoidHom G G'
      f' : MonoidHom G' G''
      inst✝² : Finite G
      inst✝¹ : IsZGroup G
      inst✝ : IsZGroup G''
      h_le : LE.le f'.ker f.range
      p : Nat
      hp : Nat.Prime p
      P : Sylow p G'
      this : Fact (Nat.Prime p)
      h_cop : (Nat.card (Subtype fun x => Membership.mem f'.ker x)).Coprime f'.ker.i …
      h : Disjoint f'.ker ↑P
      ⊢ IsCyclic (Subtype fun x => Membership.mem (↑P) x)
    -/
  · have := (P.2.map f').isCyclic_of_isZGroup
    /-
      case inr
      G : Type u_1
      G' : Type u_2
      G'' : Type u_3
      inst✝⁵ : Group G
      inst✝⁴ : Group G'
      inst✝³ : Group G''
      f : MonoidHom G G'
      f' : MonoidHom G' G''
      inst✝² : Finite G
      inst✝¹ : IsZGroup G
      inst✝ : IsZGroup G''
      h_le : LE.le f'.ker f.range
      p : Nat
      hp : Nat.Prime p
      P : Sylow p G'
      this✝ : Fact (Nat.Prime p)
      h_cop : (Nat.card (Subtype fun x => Membership.mem f'.ker x)).Coprime f'.ker.i …
      h : Disjoint f'.ker ↑P
      this : IsCyclic (Subtype fun x => Membership.mem (Subgroup.map f' ↑P) x)
      ⊢ IsCyclic (Subtype fun x => Membership.mem (↑P) x)
    -/
    apply isCyclic_of_injective (f'.subgroupMap P)
    /-
      case inr
      G : Type u_1
      G' : Type u_2
      G'' : Type u_3
      inst✝⁵ : Group G
      inst✝⁴ : Group G'
      inst✝³ : Group G''
      f : MonoidHom G G'
      f' : MonoidHom G' G''
      inst✝² : Finite G
      inst✝¹ : IsZGroup G
      inst✝ : IsZGroup G''
      h_le : LE.le f'.ker f.range
      p : Nat
      hp : Nat.Prime p
      P : Sylow p G'
      this✝ : Fact (Nat.Prime p)
      h_cop : (Nat.card (Subtype fun x => Membership.mem f'.ker x)).Coprime f'.ker.i …
      h : Disjoint f'.ker ↑P
      this : IsCyclic (Subtype fun x => Membership.mem (Subgroup.map f' ↑P) x)
      ⊢ Function.Injective ⇑(f'.subgroupMap ↑P)
    -/
    rwa [← MonoidHom.ker_eq_bot_iff, P.ker_subgroupMap f', Subgroup.subgroupOf_eq_bot]
    /-
      🎉 no goals
    -/


