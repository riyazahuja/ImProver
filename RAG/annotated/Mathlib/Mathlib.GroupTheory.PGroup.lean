/-- A p-group is a group in which every element has prime power order -/
def IsPGroup : Prop :=
  ∀ g : G, ∃ k : ℕ, g ^ p ^ k = 1


theorem iff_orderOf [hp : Fact p.Prime] : IsPGroup p G ↔ ∀ g : G, ∃ k : ℕ, orderOf g = p ^ k :=
  forall_congr' fun g =>
    ⟨fun ⟨_, hk⟩ =>
      Exists.imp (fun _ h => h.right)
        ((Nat.dvd_prime_pow hp.out).mp (orderOf_dvd_of_pow_eq_one hk)),
                                /-
                                  p : Nat
                                  G : Type u_1
                                  inst✝ : Group G
                                  hp : Fact (Nat.Prime p)
                                  g : G
                                  k : Nat
                                  hk : Eq (orderOf g) (HPow.hPow p k)
                                  ⊢ Eq (HPow.hPow g (HPow.hPow p k)) 1
                                -/
      Exists.imp fun k hk => by rw [← hk, pow_orderOf_eq_one]⟩
                                /-
                                  🎉 no goals
                                -/


theorem of_card {n : ℕ} (hG : Nat.card G = p ^ n) : IsPGroup p G := fun g =>
         /-
           p : Nat
           G : Type u_1
           inst✝ : Group G
           n : Nat
           hG : Eq (Nat.card G) (HPow.hPow p n)
           g : G
           ⊢ Eq (HPow.hPow g (HPow.hPow p n)) 1
         -/
  ⟨n, by rw [← hG, pow_card_eq_one']⟩
         /-
           🎉 no goals
         -/


theorem of_bot : IsPGroup p (⊥ : Subgroup G) :=
                       /-
                         p : Nat
                         G : Type u_1
                         inst✝ : Group G
                         ⊢ Eq (Nat.card (Subtype fun x => Membership.mem Bot.bot x)) (HPow.hPow p 0)
                       -/
  of_card (n := 0) (by rw [Subgroup.card_bot, pow_zero])
                       /-
                         🎉 no goals
                       -/


theorem iff_card [Fact p.Prime] [Finite G] : IsPGroup p G ↔ ∃ n : ℕ, Nat.card G = p ^ n := by
  /-
    p : Nat
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Finite G
    ⊢ Iff (IsPGroup p G) (Exists fun n => Eq (Nat.card G) (HPow.hPow p n))
  -/
  have hG : Nat.card G ≠ 0 := Nat.card_pos.ne'
  /-
    p : Nat
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Finite G
    hG : Ne (Nat.card G) 0
    ⊢ Iff (IsPGroup p G) (Exists fun n => Eq (Nat.card G) (HPow.hPow p n))
  -/
  refine ⟨fun h => ?_, fun ⟨n, hn⟩ => of_card hn⟩
  suffices ∀ q ∈ (Nat.card G).primeFactorsList, q = p by
    use (Nat.card G).primeFactorsList.length
    rw [← List.prod_replicate, ← List.eq_replicate_of_mem this, Nat.prod_primeFactorsList hG]
  /-
    p : Nat
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Finite G
    hG : Ne (Nat.card G) 0
    h : IsPGroup p G
    ⊢ ∀ (q : Nat), Membership.mem (Nat.card G).primeFactorsList q → Eq q p
  -/
  intro q hq
  /-
    p : Nat
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Finite G
    hG : Ne (Nat.card G) 0
    h : IsPGroup p G
    q : Nat
    hq : Membership.mem (Nat.card G).primeFactorsList q
    ⊢ Eq q p
  -/
  obtain ⟨hq1, hq2⟩ := (Nat.mem_primeFactorsList hG).mp hq
  /-
    case intro
    p : Nat
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Finite G
    hG : Ne (Nat.card G) 0
    h : IsPGroup p G
    q : Nat
    hq : Membership.mem (Nat.card G).primeFactorsList q
    hq1 : Nat.Prime q
    hq2 : Dvd.dvd q (Nat.card G)
    ⊢ Eq q p
  -/
  haveI : Fact q.Prime := ⟨hq1⟩
  /-
    case intro
    p : Nat
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Finite G
    hG : Ne (Nat.card G) 0
    h : IsPGroup p G
    q : Nat
    hq : Membership.mem (Nat.card G).primeFactorsList q
    hq1 : Nat.Prime q
    hq2 : Dvd.dvd q (Nat.card G)
    this : Fact (Nat.Prime q)
    ⊢ Eq q p
  -/
  obtain ⟨g, hg⟩ := exists_prime_orderOf_dvd_card' q hq2
  /-
    case intro.intro
    p : Nat
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Finite G
    hG : Ne (Nat.card G) 0
    h : IsPGroup p G
    q : Nat
    hq : Membership.mem (Nat.card G).primeFactorsList q
    hq1 : Nat.Prime q
    hq2 : Dvd.dvd q (Nat.card G)
    this : Fact (Nat.Prime q)
    g : G
    hg : Eq (orderOf g) q
    ⊢ Eq q p
  -/
  obtain ⟨k, hk⟩ := (iff_orderOf.mp h) g
  /-
    case intro.intro.intro
    p : Nat
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : Finite G
    hG : Ne (Nat.card G) 0
    h : IsPGroup p G
    q : Nat
    hq : Membership.mem (Nat.card G).primeFactorsList q
    hq1 : Nat.Prime q
    hq2 : Dvd.dvd q (Nat.card G)
    this : Fact (Nat.Prime q)
    g : G
    hg : Eq (orderOf g) q
    k : Nat
    hk : Eq (orderOf g) (HPow.hPow p k)
    ⊢ Eq q p
  -/
  exact (hq1.pow_eq_iff.mp (hg.symm.trans hk).symm).1.symm
  /-
    🎉 no goals
  -/


alias ⟨exists_card_eq, _⟩ := iff_card


theorem of_injective {H : Type*} [Group H] (ϕ : H →* G) (hϕ : Function.Injective ϕ) :
    IsPGroup p H := by
  /-
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    hG : IsPGroup p G
    H : Type u_2
    inst✝ : Group H
    ϕ : MonoidHom H G
    hϕ : Function.Injective ⇑ϕ
    ⊢ IsPGroup p H
  -/
  simp_rw [IsPGroup, ← hϕ.eq_iff, ϕ.map_pow, ϕ.map_one]
  /-
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    hG : IsPGroup p G
    H : Type u_2
    inst✝ : Group H
    ϕ : MonoidHom H G
    hϕ : Function.Injective ⇑ϕ
    ⊢ ∀ (g : H), Exists fun k => Eq (HPow.hPow (ϕ g) (HPow.hPow p k)) 1
  -/
  exact fun h => hG (ϕ h)
  /-
    🎉 no goals
  -/


theorem to_subgroup (H : Subgroup G) : IsPGroup p H :=
  hG.of_injective H.subtype Subtype.coe_injective


theorem of_surjective {H : Type*} [Group H] (ϕ : G →* H) (hϕ : Function.Surjective ϕ) :
    IsPGroup p H := by
  /-
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    hG : IsPGroup p G
    H : Type u_2
    inst✝ : Group H
    ϕ : MonoidHom G H
    hϕ : Function.Surjective ⇑ϕ
    ⊢ IsPGroup p H
  -/
  refine fun h => Exists.elim (hϕ h) fun g hg => Exists.imp (fun k hk => ?_) (hG g)
  /-
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    hG : IsPGroup p G
    H : Type u_2
    inst✝ : Group H
    ϕ : MonoidHom G H
    hϕ : Function.Surjective ⇑ϕ
    h : H
    g : G
    hg : Eq (ϕ g) h
    k : Nat
    hk : Eq (HPow.hPow g (HPow.hPow p k)) 1
    ⊢ Eq (HPow.hPow h (HPow.hPow p k)) 1
  -/
  rw [← hg, ← ϕ.map_pow, hk, ϕ.map_one]
  /-
    🎉 no goals
  -/


theorem to_quotient (H : Subgroup G) [H.Normal] : IsPGroup p (G ⧸ H) :=
  hG.of_surjective (QuotientGroup.mk' H) Quotient.mk''_surjective


theorem of_equiv {H : Type*} [Group H] (ϕ : G ≃* H) : IsPGroup p H :=
  hG.of_surjective ϕ.toMonoidHom ϕ.surjective


theorem orderOf_coprime {n : ℕ} (hn : p.Coprime n) (g : G) : (orderOf g).Coprime n :=
  let ⟨k, hk⟩ := hG g
  (hn.pow_left k).coprime_dvd_left (orderOf_dvd_of_pow_eq_one hk)


/-- If `gcd(p,n) = 1`, then the `n`th power map is a bijection. -/
noncomputable def powEquiv {n : ℕ} (hn : p.Coprime n) : G ≃ G :=
  let h : ∀ g : G, (Nat.card (Subgroup.zpowers g)).Coprime n := fun g =>
    (Nat.card_zpowers g).symm ▸ hG.orderOf_coprime hn g
  { toFun := (· ^ n)
    invFun := fun g => (powCoprime (h g)).symm ⟨g, Subgroup.mem_zpowers g⟩
    left_inv := fun g =>
      Subtype.ext_iff.1 <|
        (powCoprime (h (g ^ n))).left_inv
          ⟨g, _, Subtype.ext_iff.1 <| (powCoprime (h g)).left_inv ⟨g, Subgroup.mem_zpowers g⟩⟩
    right_inv := fun g =>
      Subtype.ext_iff.1 <| (powCoprime (h g)).right_inv ⟨g, Subgroup.mem_zpowers g⟩ }


@[simp]
theorem powEquiv_apply {n : ℕ} (hn : p.Coprime n) (g : G) : hG.powEquiv hn g = g ^ n :=
  rfl


@[simp]
theorem powEquiv_symm_apply {n : ℕ} (hn : p.Coprime n) (g : G) :
                                                           /-
                                                             p : Nat
                                                             G : Type u_1
                                                             inst✝ : Group G
                                                             hG : IsPGroup p G
                                                             n : Nat
                                                             hn : p.Coprime n
                                                             g : G
                                                             ⊢ Eq ((hG.powEquiv hn).symm g) (HPow.hPow g ((orderOf g).gcdB n))
                                                           -/
    (hG.powEquiv hn).symm g = g ^ (orderOf g).gcdB n := by rw [← Nat.card_zpowers]; rfl
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


/-- If `p ∤ n`, then the `n`th power map is a bijection. -/
noncomputable abbrev powEquiv' {n : ℕ} (hn : ¬p ∣ n) : G ≃ G :=
  powEquiv hG (hp.out.coprime_iff_not_dvd.mpr hn)


theorem index (H : Subgroup G) [H.FiniteIndex] : ∃ n : ℕ, H.index = p ^ n := by
  /-
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    hG : IsPGroup p G
    hp : Fact (Nat.Prime p)
    H : Subgroup G
    inst✝ : H.FiniteIndex
    ⊢ Exists fun n => Eq H.index (HPow.hPow p n)
  -/
  obtain ⟨n, hn⟩ := iff_card.mp (hG.to_quotient H.normalCore)
  obtain ⟨k, _, hk2⟩ :=
    (Nat.dvd_prime_pow hp.out).mp
      ((congr_arg _ (H.normalCore.index_eq_card.trans hn)).mp
        (Subgroup.index_dvd_of_le H.normalCore_le))
  /-
    case intro.intro.intro
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    hG : IsPGroup p G
    hp : Fact (Nat.Prime p)
    H : Subgroup G
    inst✝ : H.FiniteIndex
    n : Nat
    hn : Eq (Nat.card (HasQuotient.Quotient G H.normalCore)) (HPow.hPow p n)
    k : Nat
    left✝ : LE.le k n
    hk2 : Eq H.index (HPow.hPow p k)
    ⊢ Exists fun n => Eq H.index (HPow.hPow p n)
  -/
  exact ⟨k, hk2⟩
  /-
    🎉 no goals
  -/


theorem card_eq_or_dvd : Nat.card G = 1 ∨ p ∣ Nat.card G := by
  /-
    p : Nat
    G : Type u_1
    inst✝ : Group G
    hG : IsPGroup p G
    hp : Fact (Nat.Prime p)
    ⊢ Or (Eq (Nat.card G) 1) (Dvd.dvd p (Nat.card G))
  -/
  cases finite_or_infinite G
    /-
      case inl
      p : Nat
      G : Type u_1
      inst✝ : Group G
      hG : IsPGroup p G
      hp : Fact (Nat.Prime p)
      h✝ : Finite G
      ⊢ Or (Eq (Nat.card G) 1) (Dvd.dvd p (Nat.card G))
    -/
  · obtain ⟨n, hn⟩ := iff_card.mp hG
    /-
      case inl.intro
      p : Nat
      G : Type u_1
      inst✝ : Group G
      hG : IsPGroup p G
      hp : Fact (Nat.Prime p)
      h✝ : Finite G
      n : Nat
      hn : Eq (Nat.card G) (HPow.hPow p n)
      ⊢ Or (Eq (Nat.card G) 1) (Dvd.dvd p (Nat.card G))
    -/
    rw [hn]
    /-
      case inl.intro
      p : Nat
      G : Type u_1
      inst✝ : Group G
      hG : IsPGroup p G
      hp : Fact (Nat.Prime p)
      h✝ : Finite G
      n : Nat
      hn : Eq (Nat.card G) (HPow.hPow p n)
      ⊢ Or (Eq (HPow.hPow p n) 1) (Dvd.dvd p (HPow.hPow p n))
    -/
    cases' n with n n
      /-
        case inl.intro.zero
        p : Nat
        G : Type u_1
        inst✝ : Group G
        hG : IsPGroup p G
        hp : Fact (Nat.Prime p)
        h✝ : Finite G
        hn : Eq (Nat.card G) (HPow.hPow p 0)
        ⊢ Or (Eq (HPow.hPow p 0) 1) (Dvd.dvd p (HPow.hPow p 0))
      -/
    · exact Or.inl rfl
      /-
        🎉 no goals
      -/
      /-
        case inl.intro.succ
        p : Nat
        G : Type u_1
        inst✝ : Group G
        hG : IsPGroup p G
        hp : Fact (Nat.Prime p)
        h✝ : Finite G
        n : Nat
        hn : Eq (Nat.card G) (HPow.hPow p (HAdd.hAdd n 1))
        ⊢ Or (Eq (HPow.hPow p (HAdd.hAdd n 1)) 1) (Dvd.dvd p (HPow.hPow p (HAdd.hAdd n …
      -/
    · exact Or.inr ⟨p ^ n, by rw [pow_succ']⟩
      /-
        🎉 no goals
      -/
    /-
      case inr
      p : Nat
      G : Type u_1
      inst✝ : Group G
      hG : IsPGroup p G
      hp : Fact (Nat.Prime p)
      h✝ : Infinite G
      ⊢ Or (Eq (Nat.card G) 1) (Dvd.dvd p (Nat.card G))
    -/
  · rw [Nat.card_eq_zero_of_infinite]
    /-
      case inr
      p : Nat
      G : Type u_1
      inst✝ : Group G
      hG : IsPGroup p G
      hp : Fact (Nat.Prime p)
      h✝ : Infinite G
      ⊢ Or (Eq 0 1) (Dvd.dvd p 0)
    -/
    exact Or.inr ⟨0, rfl⟩
    /-
      🎉 no goals
    -/


theorem nontrivial_iff_card [Finite G] : Nontrivial G ↔ ∃ n > 0, Nat.card G = p ^ n :=
  ⟨fun hGnt =>
    let ⟨k, hk⟩ := iff_card.1 hG
    ⟨k,
      Nat.pos_of_ne_zero fun hk0 => by
        /-
          p : Nat
          G : Type u_1
          inst✝¹ : Group G
          hG : IsPGroup p G
          hp : Fact (Nat.Prime p)
          inst✝ : Finite G
          hGnt : Nontrivial G
          k : Nat
          hk : Eq (Nat.card G) (HPow.hPow p k)
          hk0 : Eq k 0
          ⊢ False
        -/
        rw [hk0, pow_zero] at hk; exact Finite.one_lt_card.ne' hk,
                                  /-
                                    🎉 no goals
                                  -/
      hk⟩,
    fun ⟨_, hk0, hk⟩ =>
    Finite.one_lt_card_iff_nontrivial.1 <|
      hk.symm ▸ one_lt_pow₀ (Fact.out (p := p.Prime)).one_lt (ne_of_gt hk0)⟩


theorem card_orbit (a : α) [Finite (orbit G a)] : ∃ n : ℕ, Nat.card (orbit G a) = p ^ n := by
  /-
    p : Nat
    G : Type u_1
    inst✝² : Group G
    hG : IsPGroup p G
    hp : Fact (Nat.Prime p)
    α : Type u_2
    inst✝¹ : MulAction G α
    a : α
    inst✝ : Finite ↑(MulAction.orbit G a)
    ⊢ Exists fun n => Eq (Nat.card ↑(MulAction.orbit G a)) (HPow.hPow p n)
  -/
  let ϕ := orbitEquivQuotientStabilizer G a
  /-
    p : Nat
    G : Type u_1
    inst✝² : Group G
    hG : IsPGroup p G
    hp : Fact (Nat.Prime p)
    α : Type u_2
    inst✝¹ : MulAction G α
    a : α
    inst✝ : Finite ↑(MulAction.orbit G a)
    ϕ : Equiv (↑(MulAction.orbit G a)) (HasQuotient.Quotient G (MulAction.stabiliz …
    ⊢ Exists fun n => Eq (Nat.card ↑(MulAction.orbit G a)) (HPow.hPow p n)
  -/
  haveI := Finite.of_equiv (orbit G a) ϕ
  /-
    p : Nat
    G : Type u_1
    inst✝² : Group G
    hG : IsPGroup p G
    hp : Fact (Nat.Prime p)
    α : Type u_2
    inst✝¹ : MulAction G α
    a : α
    inst✝ : Finite ↑(MulAction.orbit G a)
    ϕ : Equiv (↑(MulAction.orbit G a)) (HasQuotient.Quotient G (MulAction.stabiliz …
    this : Finite (HasQuotient.Quotient G (MulAction.stabilizer G a))
    ⊢ Exists fun n => Eq (Nat.card ↑(MulAction.orbit G a)) (HPow.hPow p n)
  -/
  haveI := (stabilizer G a).finiteIndex_of_finite_quotient
  /-
    p : Nat
    G : Type u_1
    inst✝² : Group G
    hG : IsPGroup p G
    hp : Fact (Nat.Prime p)
    α : Type u_2
    inst✝¹ : MulAction G α
    a : α
    inst✝ : Finite ↑(MulAction.orbit G a)
    ϕ : Equiv (↑(MulAction.orbit G a)) (HasQuotient.Quotient G (MulAction.stabiliz …
    this✝ : Finite (HasQuotient.Quotient G (MulAction.stabilizer G a))
    this : (MulAction.stabilizer G a).FiniteIndex
    ⊢ Exists fun n => Eq (Nat.card ↑(MulAction.orbit G a)) (HPow.hPow p n)
  -/
  rw [Nat.card_congr ϕ]
  /-
    p : Nat
    G : Type u_1
    inst✝² : Group G
    hG : IsPGroup p G
    hp : Fact (Nat.Prime p)
    α : Type u_2
    inst✝¹ : MulAction G α
    a : α
    inst✝ : Finite ↑(MulAction.orbit G a)
    ϕ : Equiv (↑(MulAction.orbit G a)) (HasQuotient.Quotient G (MulAction.stabiliz …
    this✝ : Finite (HasQuotient.Quotient G (MulAction.stabilizer G a))
    this : (MulAction.stabilizer G a).FiniteIndex
    ⊢ Exists fun n => Eq (Nat.card (HasQuotient.Quotient G (MulAction.stabilizer G …
  -/
  exact hG.index (stabilizer G a)
  /-
    🎉 no goals
  -/


/-- If `G` is a `p`-group acting on a finite set `α`, then the number of fixed points
  of the action is congruent mod `p` to the cardinality of `α` -/
theorem card_modEq_card_fixedPoints : Nat.card α ≡ Nat.card (fixedPoints G α) [MOD p] := by
  /-
    p : Nat
    G : Type u_1
    inst✝² : Group G
    hG : IsPGroup p G
    hp : Fact (Nat.Prime p)
    α : Type u_2
    inst✝¹ : MulAction G α
    inst✝ : Finite α
    ⊢ p.ModEq (Nat.card α) (Nat.card ↑(MulAction.fixedPoints G α))
  -/
  have := Fintype.ofFinite α
  /-
    p : Nat
    G : Type u_1
    inst✝² : Group G
    hG : IsPGroup p G
    hp : Fact (Nat.Prime p)
    α : Type u_2
    inst✝¹ : MulAction G α
    inst✝ : Finite α
    this : Fintype α
    ⊢ p.ModEq (Nat.card α) (Nat.card ↑(MulAction.fixedPoints G α))
  -/
  have := Fintype.ofFinite (fixedPoints G α)
  /-
    p : Nat
    G : Type u_1
    inst✝² : Group G
    hG : IsPGroup p G
    hp : Fact (Nat.Prime p)
    α : Type u_2
    inst✝¹ : MulAction G α
    inst✝ : Finite α
    this✝ : Fintype α
    this : Fintype ↑(MulAction.fixedPoints G α)
    ⊢ p.ModEq (Nat.card α) (Nat.card ↑(MulAction.fixedPoints G α))
  -/
  rw [Nat.card_eq_fintype_card, Nat.card_eq_fintype_card]
  classical
    calc
      card α = card (Σy : Quotient (orbitRel G α), { x // Quotient.mk'' x = y }) :=
        card_congr (Equiv.sigmaFiberEquiv (@Quotient.mk'' _ (orbitRel G α))).symm
      _ = ∑ a : Quotient (orbitRel G α), card { x // Quotient.mk'' x = a } := card_sigma
      _ ≡ ∑ _a : fixedPoints G α, 1 [MOD p] := ?_
      _ = _ := by simp
    rw [← ZMod.eq_iff_modEq_nat p, Nat.cast_sum, Nat.cast_sum]
    have key :
      ∀ x,
        card { y // (Quotient.mk'' y : Quotient (orbitRel G α)) = Quotient.mk'' x } =
          card (orbit G x) :=
      fun x => by simp only [Quotient.eq'']; congr
    refine
      Eq.symm
        (Finset.sum_bij_ne_zero (fun a _ _ => Quotient.mk'' a.1) (fun _ _ _ => Finset.mem_univ _)
          (fun a₁ _ _ a₂ _ _ h =>
            Subtype.eq (mem_fixedPoints'.mp a₂.2 a₁.1 (Quotient.exact' h)))
          (fun b => Quotient.inductionOn' b fun b _ hb => ?_) fun a ha _ => by
          rw [key, mem_fixedPoints_iff_card_orbit_eq_one.mp a.2])
    obtain ⟨k, hk⟩ := hG.card_orbit b
    rw [Nat.card_eq_fintype_card] at hk
    have : k = 0 := by
      contrapose! hb
      simp [-Quotient.eq, key, hk, hb]
    exact
      ⟨⟨b, mem_fixedPoints_iff_card_orbit_eq_one.2 <| by rw [hk, this, pow_zero]⟩,
        Finset.mem_univ _, ne_of_eq_of_ne Nat.cast_one one_ne_zero, rfl⟩


/-- If a p-group acts on `α` and the cardinality of `α` is not a multiple
  of `p` then the action has a fixed point. -/
theorem nonempty_fixed_point_of_prime_not_dvd_card (α) [MulAction G α] (hpα : ¬p ∣ Nat.card α) :
    (fixedPoints G α).Nonempty :=
  have : Finite α := Nat.finite_of_card_ne_zero (fun h ↦ (h ▸ hpα) (dvd_zero p))
  @Set.Nonempty.of_subtype _ _
    (by
      /-
        p : Nat
        G : Type u_1
        inst✝¹ : Group G
        hG : IsPGroup p G
        hp : Fact (Nat.Prime p)
        α : Type u_3
        inst✝ : MulAction G α
        hpα : Not (Dvd.dvd p (Nat.card α))
        this : Finite α
        ⊢ Nonempty ↑(MulAction.fixedPoints G α)
      -/
      rw [← Finite.card_pos_iff, pos_iff_ne_zero]
      /-
        p : Nat
        G : Type u_1
        inst✝¹ : Group G
        hG : IsPGroup p G
        hp : Fact (Nat.Prime p)
        α : Type u_3
        inst✝ : MulAction G α
        hpα : Not (Dvd.dvd p (Nat.card α))
        this : Finite α
        ⊢ Ne (Nat.card ↑(MulAction.fixedPoints G α)) 0
      -/
      contrapose! hpα
      /-
        p : Nat
        G : Type u_1
        inst✝¹ : Group G
        hG : IsPGroup p G
        hp : Fact (Nat.Prime p)
        α : Type u_3
        inst✝ : MulAction G α
        this : Finite α
        hpα : Eq (Nat.card ↑(MulAction.fixedPoints G α)) 0
        ⊢ Dvd.dvd p (Nat.card α)
      -/
      rw [← Nat.modEq_zero_iff_dvd, ← hpα]
      /-
        p : Nat
        G : Type u_1
        inst✝¹ : Group G
        hG : IsPGroup p G
        hp : Fact (Nat.Prime p)
        α : Type u_3
        inst✝ : MulAction G α
        this : Finite α
        hpα : Eq (Nat.card ↑(MulAction.fixedPoints G α)) 0
        ⊢ p.ModEq (Nat.card α) (Nat.card ↑(MulAction.fixedPoints G α))
      -/
      exact hG.card_modEq_card_fixedPoints α)
      /-
        🎉 no goals
      -/


/-- If a p-group acts on `α` and the cardinality of `α` is a multiple
  of `p`, and the action has one fixed point, then it has another fixed point. -/
theorem exists_fixed_point_of_prime_dvd_card_of_fixed_point (hpα : p ∣ Nat.card α) {a : α}
    (ha : a ∈ fixedPoints G α) : ∃ b, b ∈ fixedPoints G α ∧ a ≠ b := by
  have hpf : p ∣ Nat.card (fixedPoints G α) :=
    Nat.modEq_zero_iff_dvd.mp ((hG.card_modEq_card_fixedPoints α).symm.trans hpα.modEq_zero_nat)
  have hα : 1 < Nat.card (fixedPoints G α) :=
    (Fact.out (p := p.Prime)).one_lt.trans_le (Nat.le_of_dvd (Finite.card_pos_iff.2 ⟨⟨a, ha⟩⟩) hpf)
  /-
    p : Nat
    G : Type u_1
    inst✝² : Group G
    hG : IsPGroup p G
    hp : Fact (Nat.Prime p)
    α : Type u_2
    inst✝¹ : MulAction G α
    inst✝ : Finite α
    hpα : Dvd.dvd p (Nat.card α)
    a : α
    ha : Membership.mem (MulAction.fixedPoints G α) a
    hpf : Dvd.dvd p (Nat.card ↑(MulAction.fixedPoints G α))
    hα : LT.lt 1 (Nat.card ↑(MulAction.fixedPoints G α))
    ⊢ Exists fun b => And (Membership.mem (MulAction.fixedPoints G α) b) (Ne a b)
  -/
  rw [Finite.one_lt_card_iff_nontrivial] at hα
  exact
    let ⟨⟨b, hb⟩, hba⟩ := exists_ne (⟨a, ha⟩ : fixedPoints G α)
    ⟨b, hb, fun hab => hba (by simp_rw [hab])⟩


theorem center_nontrivial [Nontrivial G] [Finite G] : Nontrivial (Subgroup.center G) := by
  classical
    have := (hG.of_equiv ConjAct.toConjAct).exists_fixed_point_of_prime_dvd_card_of_fixed_point G
    rw [ConjAct.fixedPoints_eq_center] at this
    have dvd : p ∣ Nat.card G := by
      obtain ⟨n, hn0, hn⟩ := hG.nontrivial_iff_card.mp inferInstance
      exact hn.symm ▸ dvd_pow_self _ (ne_of_gt hn0)
    obtain ⟨g, hg⟩ := this dvd (Subgroup.center G).one_mem
    exact ⟨⟨1, ⟨g, hg.1⟩, mt Subtype.ext_iff.mp hg.2⟩⟩


theorem bot_lt_center [Nontrivial G] [Finite G] : ⊥ < Subgroup.center G := by
  /-
    p : Nat
    G : Type u_1
    inst✝² : Group G
    hG : IsPGroup p G
    hp : Fact (Nat.Prime p)
    inst✝¹ : Nontrivial G
    inst✝ : Finite G
    ⊢ LT.lt Bot.bot (Subgroup.center G)
  -/
  haveI := center_nontrivial hG
  classical exact
      bot_lt_iff_ne_bot.mpr ((Subgroup.center G).one_lt_card_iff_ne_bot.mp Finite.one_lt_card)


theorem to_le {H K : Subgroup G} (hK : IsPGroup p K) (hHK : H ≤ K) : IsPGroup p H :=
  hK.of_injective (Subgroup.inclusion hHK) fun a b h =>
    Subtype.ext (by
      /-
        p : Nat
        G : Type u_1
        inst✝ : Group G
        H K : Subgroup G
        hK : IsPGroup p (Subtype fun x => Membership.mem K x)
        hHK : LE.le H K
        a b : Subtype fun x => Membership.mem H x
        h : Eq ((Subgroup.inclusion hHK) a) ((Subgroup.inclusion hHK) b)
        ⊢ Eq ↑a ↑b
      -/
      change ((Subgroup.inclusion hHK) a : G) = (Subgroup.inclusion hHK) b
      /-
        p : Nat
        G : Type u_1
        inst✝ : Group G
        H K : Subgroup G
        hK : IsPGroup p (Subtype fun x => Membership.mem K x)
        hHK : LE.le H K
        a b : Subtype fun x => Membership.mem H x
        h : Eq ((Subgroup.inclusion hHK) a) ((Subgroup.inclusion hHK) b)
        ⊢ Eq ↑((Subgroup.inclusion hHK) a) ↑((Subgroup.inclusion hHK) b)
      -/
      apply Subtype.ext_iff.mp h)
      /-
        🎉 no goals
      -/


theorem to_inf_left {H K : Subgroup G} (hH : IsPGroup p H) : IsPGroup p (H ⊓ K : Subgroup G) :=
  hH.to_le inf_le_left


theorem to_inf_right {H K : Subgroup G} (hK : IsPGroup p K) : IsPGroup p (H ⊓ K : Subgroup G) :=
  hK.to_le inf_le_right


theorem map {H : Subgroup G} (hH : IsPGroup p H) {K : Type*} [Group K] (ϕ : G →* K) :
    IsPGroup p (H.map ϕ) := by
  /-
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    hH : IsPGroup p (Subtype fun x => Membership.mem H x)
    K : Type u_2
    inst✝ : Group K
    ϕ : MonoidHom G K
    ⊢ IsPGroup p (Subtype fun x => Membership.mem (Subgroup.map ϕ H) x)
  -/
  rw [← H.range_subtype, MonoidHom.map_range]
  /-
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    hH : IsPGroup p (Subtype fun x => Membership.mem H x)
    K : Type u_2
    inst✝ : Group K
    ϕ : MonoidHom G K
    ⊢ IsPGroup p (Subtype fun x => Membership.mem (ϕ.comp H.subtype).range x)
  -/
  exact hH.of_surjective (ϕ.restrict H).rangeRestrict (ϕ.restrict H).rangeRestrict_surjective
  /-
    🎉 no goals
  -/


theorem comap_of_ker_isPGroup {H : Subgroup G} (hH : IsPGroup p H) {K : Type*} [Group K]
    (ϕ : K →* G) (hϕ : IsPGroup p ϕ.ker) : IsPGroup p (H.comap ϕ) := by
  /-
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    hH : IsPGroup p (Subtype fun x => Membership.mem H x)
    K : Type u_2
    inst✝ : Group K
    ϕ : MonoidHom K G
    hϕ : IsPGroup p (Subtype fun x => Membership.mem ϕ.ker x)
    ⊢ IsPGroup p (Subtype fun x => Membership.mem (Subgroup.comap ϕ H) x)
  -/
  intro g
  /-
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    hH : IsPGroup p (Subtype fun x => Membership.mem H x)
    K : Type u_2
    inst✝ : Group K
    ϕ : MonoidHom K G
    hϕ : IsPGroup p (Subtype fun x => Membership.mem ϕ.ker x)
    g : Subtype fun x => Membership.mem (Subgroup.comap ϕ H) x
    ⊢ Exists fun k => Eq (HPow.hPow g (HPow.hPow p k)) 1
  -/
  obtain ⟨j, hj⟩ := hH ⟨ϕ g.1, g.2⟩
  /-
    case intro
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    hH : IsPGroup p (Subtype fun x => Membership.mem H x)
    K : Type u_2
    inst✝ : Group K
    ϕ : MonoidHom K G
    hϕ : IsPGroup p (Subtype fun x => Membership.mem ϕ.ker x)
    g : Subtype fun x => Membership.mem (Subgroup.comap ϕ H) x
    j : Nat
    hj : Eq (HPow.hPow ⟨ϕ ↑g, ⋯⟩ (HPow.hPow p j)) 1
    ⊢ Exists fun k => Eq (HPow.hPow g (HPow.hPow p k)) 1
  -/
  rw [Subtype.ext_iff, H.coe_pow, Subtype.coe_mk, ← ϕ.map_pow] at hj
  /-
    case intro
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    hH : IsPGroup p (Subtype fun x => Membership.mem H x)
    K : Type u_2
    inst✝ : Group K
    ϕ : MonoidHom K G
    hϕ : IsPGroup p (Subtype fun x => Membership.mem ϕ.ker x)
    g : Subtype fun x => Membership.mem (Subgroup.comap ϕ H) x
    j : Nat
    hj : Eq (ϕ (HPow.hPow (↑g) (HPow.hPow p j))) ↑1
    ⊢ Exists fun k => Eq (HPow.hPow g (HPow.hPow p k)) 1
  -/
  obtain ⟨k, hk⟩ := hϕ ⟨g.1 ^ p ^ j, hj⟩
  /-
    case intro.intro
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    hH : IsPGroup p (Subtype fun x => Membership.mem H x)
    K : Type u_2
    inst✝ : Group K
    ϕ : MonoidHom K G
    hϕ : IsPGroup p (Subtype fun x => Membership.mem ϕ.ker x)
    g : Subtype fun x => Membership.mem (Subgroup.comap ϕ H) x
    j : Nat
    hj : Eq (ϕ (HPow.hPow (↑g) (HPow.hPow p j))) ↑1
    k : Nat
    hk : Eq (HPow.hPow ⟨HPow.hPow (↑g) (HPow.hPow p j), hj⟩ (HPow.hPow p k)) 1
    ⊢ Exists fun k => Eq (HPow.hPow g (HPow.hPow p k)) 1
  -/
  rw [Subtype.ext_iff, ϕ.ker.coe_pow, Subtype.coe_mk, ← pow_mul, ← pow_add] at hk
  /-
    case intro.intro
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    hH : IsPGroup p (Subtype fun x => Membership.mem H x)
    K : Type u_2
    inst✝ : Group K
    ϕ : MonoidHom K G
    hϕ : IsPGroup p (Subtype fun x => Membership.mem ϕ.ker x)
    g : Subtype fun x => Membership.mem (Subgroup.comap ϕ H) x
    j : Nat
    hj : Eq (ϕ (HPow.hPow (↑g) (HPow.hPow p j))) ↑1
    k : Nat
    hk : Eq (HPow.hPow (↑g) (HPow.hPow p (HAdd.hAdd j k))) ↑1
    ⊢ Exists fun k => Eq (HPow.hPow g (HPow.hPow p k)) 1
  -/
  exact ⟨j + k, by rwa [Subtype.ext_iff, (H.comap ϕ).coe_pow]⟩
  /-
    🎉 no goals
  -/


theorem ker_isPGroup_of_injective {K : Type*} [Group K] {ϕ : K →* G} (hϕ : Function.Injective ϕ) :
    IsPGroup p ϕ.ker :=
  (congr_arg (fun Q : Subgroup K => IsPGroup p Q) (ϕ.ker_eq_bot_iff.mpr hϕ)).mpr IsPGroup.of_bot


theorem comap_of_injective {H : Subgroup G} (hH : IsPGroup p H) {K : Type*} [Group K] (ϕ : K →* G)
    (hϕ : Function.Injective ϕ) : IsPGroup p (H.comap ϕ) :=
  hH.comap_of_ker_isPGroup ϕ (ker_isPGroup_of_injective hϕ)


theorem comap_subtype {H : Subgroup G} (hH : IsPGroup p H) {K : Subgroup G} :
    IsPGroup p (H.comap K.subtype) :=
  hH.comap_of_injective K.subtype Subtype.coe_injective


theorem to_sup_of_normal_right {H K : Subgroup G} (hH : IsPGroup p H) (hK : IsPGroup p K)
    [K.Normal] : IsPGroup p (H ⊔ K : Subgroup G) := by
  /-
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    H K : Subgroup G
    hH : IsPGroup p (Subtype fun x => Membership.mem H x)
    hK : IsPGroup p (Subtype fun x => Membership.mem K x)
    inst✝ : K.Normal
    ⊢ IsPGroup p (Subtype fun x => Membership.mem (Max.max H K) x)
  -/
  rw [← QuotientGroup.ker_mk' K, ← Subgroup.comap_map_eq]
  /-
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    H K : Subgroup G
    hH : IsPGroup p (Subtype fun x => Membership.mem H x)
    hK : IsPGroup p (Subtype fun x => Membership.mem K x)
    inst✝ : K.Normal
    ⊢ IsPGroup p (Subtype fun x => Membership.mem (Subgroup.comap (QuotientGroup.m …
  -/
  apply (hH.map (QuotientGroup.mk' K)).comap_of_ker_isPGroup
  /-
    case hϕ
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    H K : Subgroup G
    hH : IsPGroup p (Subtype fun x => Membership.mem H x)
    hK : IsPGroup p (Subtype fun x => Membership.mem K x)
    inst✝ : K.Normal
    ⊢ IsPGroup p (Subtype fun x => Membership.mem (QuotientGroup.mk' K).ker x)
  -/
  rwa [QuotientGroup.ker_mk']
  /-
    🎉 no goals
  -/


theorem to_sup_of_normal_left {H K : Subgroup G} (hH : IsPGroup p H) (hK : IsPGroup p K)
    [H.Normal] : IsPGroup p (H ⊔ K : Subgroup G) := sup_comm H K ▸ to_sup_of_normal_right hK hH


theorem to_sup_of_normal_right' {H K : Subgroup G} (hH : IsPGroup p H) (hK : IsPGroup p K)
    (hHK : H ≤ K.normalizer) : IsPGroup p (H ⊔ K : Subgroup G) :=
  let hHK' :=
    to_sup_of_normal_right (hH.of_equiv (Subgroup.subgroupOfEquivOfLe hHK).symm)
      (hK.of_equiv (Subgroup.subgroupOfEquivOfLe Subgroup.le_normalizer).symm)
  ((congr_arg (fun H : Subgroup K.normalizer => IsPGroup p H)
            (Subgroup.sup_subgroupOf_eq hHK Subgroup.le_normalizer)).mp
        hHK').of_equiv
    (Subgroup.subgroupOfEquivOfLe (sup_le hHK Subgroup.le_normalizer))


theorem to_sup_of_normal_left' {H K : Subgroup G} (hH : IsPGroup p H) (hK : IsPGroup p K)
    (hHK : K ≤ H.normalizer) : IsPGroup p (H ⊔ K : Subgroup G) :=
  sup_comm H K ▸ to_sup_of_normal_right' hK hH hHK


/-- finite p-groups with different p have coprime orders -/
theorem coprime_card_of_ne {G₂ : Type*} [Group G₂] (p₁ p₂ : ℕ) [hp₁ : Fact p₁.Prime]
    [hp₂ : Fact p₂.Prime] (hne : p₁ ≠ p₂) (H₁ : Subgroup G) (H₂ : Subgroup G₂) [Finite H₁]
    [Finite H₂] (hH₁ : IsPGroup p₁ H₁) (hH₂ : IsPGroup p₂ H₂) :
    Nat.Coprime (Nat.card H₁) (Nat.card H₂) := by
  /-
    G : Type u_1
    inst✝³ : Group G
    G₂ : Type u_2
    inst✝² : Group G₂
    p₁ p₂ : Nat
    hp₁ : Fact (Nat.Prime p₁)
    hp₂ : Fact (Nat.Prime p₂)
    hne : Ne p₁ p₂
    H₁ : Subgroup G
    H₂ : Subgroup G₂
    inst✝¹ : Finite (Subtype fun x => Membership.mem H₁ x)
    inst✝ : Finite (Subtype fun x => Membership.mem H₂ x)
    hH₁ : IsPGroup p₁ (Subtype fun x => Membership.mem H₁ x)
    hH₂ : IsPGroup p₂ (Subtype fun x => Membership.mem H₂ x)
    ⊢ (Nat.card (Subtype fun x => Membership.mem H₁ x)).Coprime (Nat.card (Subtype …
  -/
  obtain ⟨n₁, heq₁⟩ := iff_card.mp hH₁; rw [heq₁]; clear heq₁
  /-
    case intro
    G : Type u_1
    inst✝³ : Group G
    G₂ : Type u_2
    inst✝² : Group G₂
    p₁ p₂ : Nat
    hp₁ : Fact (Nat.Prime p₁)
    hp₂ : Fact (Nat.Prime p₂)
    hne : Ne p₁ p₂
    H₁ : Subgroup G
    H₂ : Subgroup G₂
    inst✝¹ : Finite (Subtype fun x => Membership.mem H₁ x)
    inst✝ : Finite (Subtype fun x => Membership.mem H₂ x)
    hH₁ : IsPGroup p₁ (Subtype fun x => Membership.mem H₁ x)
    hH₂ : IsPGroup p₂ (Subtype fun x => Membership.mem H₂ x)
    n₁ : Nat
    ⊢ (HPow.hPow p₁ n₁).Coprime (Nat.card (Subtype fun x => Membership.mem H₂ x))
  -/
  obtain ⟨n₂, heq₂⟩ := iff_card.mp hH₂; rw [heq₂]; clear heq₂
  /-
    case intro.intro
    G : Type u_1
    inst✝³ : Group G
    G₂ : Type u_2
    inst✝² : Group G₂
    p₁ p₂ : Nat
    hp₁ : Fact (Nat.Prime p₁)
    hp₂ : Fact (Nat.Prime p₂)
    hne : Ne p₁ p₂
    H₁ : Subgroup G
    H₂ : Subgroup G₂
    inst✝¹ : Finite (Subtype fun x => Membership.mem H₁ x)
    inst✝ : Finite (Subtype fun x => Membership.mem H₂ x)
    hH₁ : IsPGroup p₁ (Subtype fun x => Membership.mem H₁ x)
    hH₂ : IsPGroup p₂ (Subtype fun x => Membership.mem H₂ x)
    n₁ n₂ : Nat
    ⊢ (HPow.hPow p₁ n₁).Coprime (HPow.hPow p₂ n₂)
  -/
  exact Nat.coprime_pow_primes _ _ hp₁.elim hp₂.elim hne
  /-
    🎉 no goals
  -/


/-- p-groups with different p are disjoint -/
theorem disjoint_of_ne (p₁ p₂ : ℕ) [hp₁ : Fact p₁.Prime] [hp₂ : Fact p₂.Prime] (hne : p₁ ≠ p₂)
    (H₁ H₂ : Subgroup G) (hH₁ : IsPGroup p₁ H₁) (hH₂ : IsPGroup p₂ H₂) : Disjoint H₁ H₂ := by
  /-
    G : Type u_1
    inst✝ : Group G
    p₁ p₂ : Nat
    hp₁ : Fact (Nat.Prime p₁)
    hp₂ : Fact (Nat.Prime p₂)
    hne : Ne p₁ p₂
    H₁ H₂ : Subgroup G
    hH₁ : IsPGroup p₁ (Subtype fun x => Membership.mem H₁ x)
    hH₂ : IsPGroup p₂ (Subtype fun x => Membership.mem H₂ x)
    ⊢ Disjoint H₁ H₂
  -/
  rw [Subgroup.disjoint_def]
  /-
    G : Type u_1
    inst✝ : Group G
    p₁ p₂ : Nat
    hp₁ : Fact (Nat.Prime p₁)
    hp₂ : Fact (Nat.Prime p₂)
    hne : Ne p₁ p₂
    H₁ H₂ : Subgroup G
    hH₁ : IsPGroup p₁ (Subtype fun x => Membership.mem H₁ x)
    hH₂ : IsPGroup p₂ (Subtype fun x => Membership.mem H₂ x)
    ⊢ ∀ {x : G}, Membership.mem H₁ x → Membership.mem H₂ x → Eq x 1
  -/
  intro x hx₁ hx₂
  /-
    G : Type u_1
    inst✝ : Group G
    p₁ p₂ : Nat
    hp₁ : Fact (Nat.Prime p₁)
    hp₂ : Fact (Nat.Prime p₂)
    hne : Ne p₁ p₂
    H₁ H₂ : Subgroup G
    hH₁ : IsPGroup p₁ (Subtype fun x => Membership.mem H₁ x)
    hH₂ : IsPGroup p₂ (Subtype fun x => Membership.mem H₂ x)
    x : G
    hx₁ : Membership.mem H₁ x
    hx₂ : Membership.mem H₂ x
    ⊢ Eq x 1
  -/
  obtain ⟨n₁, hn₁⟩ := iff_orderOf.mp hH₁ ⟨x, hx₁⟩
  /-
    case intro
    G : Type u_1
    inst✝ : Group G
    p₁ p₂ : Nat
    hp₁ : Fact (Nat.Prime p₁)
    hp₂ : Fact (Nat.Prime p₂)
    hne : Ne p₁ p₂
    H₁ H₂ : Subgroup G
    hH₁ : IsPGroup p₁ (Subtype fun x => Membership.mem H₁ x)
    hH₂ : IsPGroup p₂ (Subtype fun x => Membership.mem H₂ x)
    x : G
    hx₁ : Membership.mem H₁ x
    hx₂ : Membership.mem H₂ x
    n₁ : Nat
    hn₁ : Eq (orderOf ⟨x, hx₁⟩) (HPow.hPow p₁ n₁)
    ⊢ Eq x 1
  -/
  obtain ⟨n₂, hn₂⟩ := iff_orderOf.mp hH₂ ⟨x, hx₂⟩
  /-
    case intro.intro
    G : Type u_1
    inst✝ : Group G
    p₁ p₂ : Nat
    hp₁ : Fact (Nat.Prime p₁)
    hp₂ : Fact (Nat.Prime p₂)
    hne : Ne p₁ p₂
    H₁ H₂ : Subgroup G
    hH₁ : IsPGroup p₁ (Subtype fun x => Membership.mem H₁ x)
    hH₂ : IsPGroup p₂ (Subtype fun x => Membership.mem H₂ x)
    x : G
    hx₁ : Membership.mem H₁ x
    hx₂ : Membership.mem H₂ x
    n₁ : Nat
    hn₁ : Eq (orderOf ⟨x, hx₁⟩) (HPow.hPow p₁ n₁)
    n₂ : Nat
    hn₂ : Eq (orderOf ⟨x, hx₂⟩) (HPow.hPow p₂ n₂)
    ⊢ Eq x 1
  -/
  rw [Subgroup.orderOf_mk] at hn₁ hn₂
  /-
    case intro.intro
    G : Type u_1
    inst✝ : Group G
    p₁ p₂ : Nat
    hp₁ : Fact (Nat.Prime p₁)
    hp₂ : Fact (Nat.Prime p₂)
    hne : Ne p₁ p₂
    H₁ H₂ : Subgroup G
    hH₁ : IsPGroup p₁ (Subtype fun x => Membership.mem H₁ x)
    hH₂ : IsPGroup p₂ (Subtype fun x => Membership.mem H₂ x)
    x : G
    hx₁ : Membership.mem H₁ x
    hx₂ : Membership.mem H₂ x
    n₁ : Nat
    hn₁ : Eq (orderOf x) (HPow.hPow p₁ n₁)
    n₂ : Nat
    hn₂ : Eq (orderOf x) (HPow.hPow p₂ n₂)
    ⊢ Eq x 1
  -/
  have : p₁ ^ n₁ = p₂ ^ n₂ := by rw [← hn₁, ← hn₂]
  /-
    case intro.intro
    G : Type u_1
    inst✝ : Group G
    p₁ p₂ : Nat
    hp₁ : Fact (Nat.Prime p₁)
    hp₂ : Fact (Nat.Prime p₂)
    hne : Ne p₁ p₂
    H₁ H₂ : Subgroup G
    hH₁ : IsPGroup p₁ (Subtype fun x => Membership.mem H₁ x)
    hH₂ : IsPGroup p₂ (Subtype fun x => Membership.mem H₂ x)
    x : G
    hx₁ : Membership.mem H₁ x
    hx₂ : Membership.mem H₂ x
    n₁ : Nat
    hn₁ : Eq (orderOf x) (HPow.hPow p₁ n₁)
    n₂ : Nat
    hn₂ : Eq (orderOf x) (HPow.hPow p₂ n₂)
    this : Eq (HPow.hPow p₁ n₁) (HPow.hPow p₂ n₂)
    ⊢ Eq x 1
  -/
  rcases n₁.eq_zero_or_pos with (rfl | hn₁)
    /-
      case intro.intro.inl
      G : Type u_1
      inst✝ : Group G
      p₁ p₂ : Nat
      hp₁ : Fact (Nat.Prime p₁)
      hp₂ : Fact (Nat.Prime p₂)
      hne : Ne p₁ p₂
      H₁ H₂ : Subgroup G
      hH₁ : IsPGroup p₁ (Subtype fun x => Membership.mem H₁ x)
      hH₂ : IsPGroup p₂ (Subtype fun x => Membership.mem H₂ x)
      x : G
      hx₁ : Membership.mem H₁ x
      hx₂ : Membership.mem H₂ x
      n₂ : Nat
      hn₂ : Eq (orderOf x) (HPow.hPow p₂ n₂)
      hn₁ : Eq (orderOf x) (HPow.hPow p₁ 0)
      this : Eq (HPow.hPow p₁ 0) (HPow.hPow p₂ n₂)
      ⊢ Eq x 1
    -/
  · simpa using hn₁
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.inr
      G : Type u_1
      inst✝ : Group G
      p₁ p₂ : Nat
      hp₁ : Fact (Nat.Prime p₁)
      hp₂ : Fact (Nat.Prime p₂)
      hne : Ne p₁ p₂
      H₁ H₂ : Subgroup G
      hH₁ : IsPGroup p₁ (Subtype fun x => Membership.mem H₁ x)
      hH₂ : IsPGroup p₂ (Subtype fun x => Membership.mem H₂ x)
      x : G
      hx₁ : Membership.mem H₁ x
      hx₂ : Membership.mem H₂ x
      n₁ : Nat
      hn₁✝ : Eq (orderOf x) (HPow.hPow p₁ n₁)
      n₂ : Nat
      hn₂ : Eq (orderOf x) (HPow.hPow p₂ n₂)
      this : Eq (HPow.hPow p₁ n₁) (HPow.hPow p₂ n₂)
      hn₁ : GT.gt n₁ 0
      ⊢ Eq x 1
    -/
  · exact absurd (eq_of_prime_pow_eq hp₁.out.prime hp₂.out.prime hn₁ this) hne
    /-
      🎉 no goals
    -/


theorem le_or_disjoint_of_coprime [hp : Fact p.Prime] {P : Subgroup G} (hP : IsPGroup p P)
    {H : Subgroup G} [H.Normal] (h_cop : (Nat.card H).Coprime H.index) :
    P ≤ H ∨ Disjoint H P := by
  /-
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    hp : Fact (Nat.Prime p)
    P : Subgroup G
    hP : IsPGroup p (Subtype fun x => Membership.mem P x)
    H : Subgroup G
    inst✝ : H.Normal
    h_cop : (Nat.card (Subtype fun x => Membership.mem H x)).Coprime H.index
    ⊢ Or (LE.le P H) (Disjoint H P)
  -/
  by_cases h1 : Nat.card H = 0
    /-
      case pos
      p : Nat
      G : Type u_1
      inst✝¹ : Group G
      hp : Fact (Nat.Prime p)
      P : Subgroup G
      hP : IsPGroup p (Subtype fun x => Membership.mem P x)
      H : Subgroup G
      inst✝ : H.Normal
      h_cop : (Nat.card (Subtype fun x => Membership.mem H x)).Coprime H.index
      h1 : Eq (Nat.card (Subtype fun x => Membership.mem H x)) 0
      ⊢ Or (LE.le P H) (Disjoint H P)
    -/
  · rw [h1, Nat.coprime_zero_left, Subgroup.index_eq_one] at h_cop
    /-
      case pos
      p : Nat
      G : Type u_1
      inst✝¹ : Group G
      hp : Fact (Nat.Prime p)
      P : Subgroup G
      hP : IsPGroup p (Subtype fun x => Membership.mem P x)
      H : Subgroup G
      inst✝ : H.Normal
      h_cop : Eq H Top.top
      h1 : Eq (Nat.card (Subtype fun x => Membership.mem H x)) 0
      ⊢ Or (LE.le P H) (Disjoint H P)
    -/
    rw [h_cop]
    /-
      case pos
      p : Nat
      G : Type u_1
      inst✝¹ : Group G
      hp : Fact (Nat.Prime p)
      P : Subgroup G
      hP : IsPGroup p (Subtype fun x => Membership.mem P x)
      H : Subgroup G
      inst✝ : H.Normal
      h_cop : Eq H Top.top
      h1 : Eq (Nat.card (Subtype fun x => Membership.mem H x)) 0
      ⊢ Or (LE.le P Top.top) (Disjoint Top.top P)
    -/
    exact Or.inl le_top
    /-
      🎉 no goals
    -/
  /-
    case neg
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    hp : Fact (Nat.Prime p)
    P : Subgroup G
    hP : IsPGroup p (Subtype fun x => Membership.mem P x)
    H : Subgroup G
    inst✝ : H.Normal
    h_cop : (Nat.card (Subtype fun x => Membership.mem H x)).Coprime H.index
    h1 : Not (Eq (Nat.card (Subtype fun x => Membership.mem H x)) 0)
    ⊢ Or (LE.le P H) (Disjoint H P)
  -/
  by_cases h2 : H.index = 0
    /-
      case pos
      p : Nat
      G : Type u_1
      inst✝¹ : Group G
      hp : Fact (Nat.Prime p)
      P : Subgroup G
      hP : IsPGroup p (Subtype fun x => Membership.mem P x)
      H : Subgroup G
      inst✝ : H.Normal
      h_cop : (Nat.card (Subtype fun x => Membership.mem H x)).Coprime H.index
      h1 : Not (Eq (Nat.card (Subtype fun x => Membership.mem H x)) 0)
      h2 : Eq H.index 0
      ⊢ Or (LE.le P H) (Disjoint H P)
    -/
  · rw [h2, Nat.coprime_zero_right, Subgroup.card_eq_one] at h_cop
    /-
      case pos
      p : Nat
      G : Type u_1
      inst✝¹ : Group G
      hp : Fact (Nat.Prime p)
      P : Subgroup G
      hP : IsPGroup p (Subtype fun x => Membership.mem P x)
      H : Subgroup G
      inst✝ : H.Normal
      h_cop : Eq H Bot.bot
      h1 : Not (Eq (Nat.card (Subtype fun x => Membership.mem H x)) 0)
      h2 : Eq H.index 0
      ⊢ Or (LE.le P H) (Disjoint H P)
    -/
    rw [h_cop]
    /-
      case pos
      p : Nat
      G : Type u_1
      inst✝¹ : Group G
      hp : Fact (Nat.Prime p)
      P : Subgroup G
      hP : IsPGroup p (Subtype fun x => Membership.mem P x)
      H : Subgroup G
      inst✝ : H.Normal
      h_cop : Eq H Bot.bot
      h1 : Not (Eq (Nat.card (Subtype fun x => Membership.mem H x)) 0)
      h2 : Eq H.index 0
      ⊢ Or (LE.le P Bot.bot) (Disjoint Bot.bot P)
    -/
    exact Or.inr disjoint_bot_left
    /-
      🎉 no goals
    -/
  have : Finite G := by
    apply Nat.finite_of_card_ne_zero
    rw [← H.card_mul_index]
    exact mul_ne_zero h1 h2
  have h3 : (Nat.card H).Coprime (Nat.card P) ∨ H.index.Coprime (Nat.card P) := by
    obtain ⟨k, hk⟩ := hP.exists_card_eq
    refine hk ▸ Or.imp hp.out.coprime_pow_of_not_dvd hp.out.coprime_pow_of_not_dvd ?_
    contrapose! h_cop
    exact Nat.Prime.not_coprime_iff_dvd.mpr ⟨p, hp.out, h_cop⟩
  /-
    case neg
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    hp : Fact (Nat.Prime p)
    P : Subgroup G
    hP : IsPGroup p (Subtype fun x => Membership.mem P x)
    H : Subgroup G
    inst✝ : H.Normal
    h_cop : (Nat.card (Subtype fun x => Membership.mem H x)).Coprime H.index
    h1 : Not (Eq (Nat.card (Subtype fun x => Membership.mem H x)) 0)
    h2 : Not (Eq H.index 0)
    this : Finite G
    h3 : Or ((Nat.card (Subtype fun x => Membership.mem H x)).Coprime (Nat.card (S …
    ⊢ Or (LE.le P H) (Disjoint H P)
  -/
  refine h3.symm.imp (fun h4 ↦ ?_) (fun h4 ↦ ?_)
    /-
      case neg.refine_1
      p : Nat
      G : Type u_1
      inst✝¹ : Group G
      hp : Fact (Nat.Prime p)
      P : Subgroup G
      hP : IsPGroup p (Subtype fun x => Membership.mem P x)
      H : Subgroup G
      inst✝ : H.Normal
      h_cop : (Nat.card (Subtype fun x => Membership.mem H x)).Coprime H.index
      h1 : Not (Eq (Nat.card (Subtype fun x => Membership.mem H x)) 0)
      h2 : Not (Eq H.index 0)
      this : Finite G
      h3 : Or ((Nat.card (Subtype fun x => Membership.mem H x)).Coprime (Nat.card (S …
      h4 : H.index.Coprime (Nat.card (Subtype fun x => Membership.mem P x))
      ⊢ LE.le P H
    -/
  · rw [← Subgroup.relindex_eq_one]
    exact Nat.eq_one_of_dvd_coprimes h4 (H.relindex_dvd_index_of_normal P)
      (Subgroup.relindex_dvd_card H P)
    /-
      case neg.refine_2
      p : Nat
      G : Type u_1
      inst✝¹ : Group G
      hp : Fact (Nat.Prime p)
      P : Subgroup G
      hP : IsPGroup p (Subtype fun x => Membership.mem P x)
      H : Subgroup G
      inst✝ : H.Normal
      h_cop : (Nat.card (Subtype fun x => Membership.mem H x)).Coprime H.index
      h1 : Not (Eq (Nat.card (Subtype fun x => Membership.mem H x)) 0)
      h2 : Not (Eq H.index 0)
      this : Finite G
      h3 : Or ((Nat.card (Subtype fun x => Membership.mem H x)).Coprime (Nat.card (S …
      h4 : (Nat.card (Subtype fun x => Membership.mem H x)).Coprime (Nat.card (Subty …
      ⊢ Disjoint H P
    -/
  · exact disjoint_iff.mpr (Subgroup.inf_eq_bot_of_coprime h4)
    /-
      🎉 no goals
    -/


/-- The cardinality of the `center` of a `p`-group is `p ^ k` where `k` is positive. -/
theorem card_center_eq_prime_pow (hGpn : Nat.card G = p ^ n) (hn : 0 < n) :
    ∃ k > 0, Nat.card (center G) = p ^ k := by
  /-
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Fact (Nat.Prime p)
    n : Nat
    hGpn : Eq (Nat.card G) (HPow.hPow p n)
    hn : LT.lt 0 n
    ⊢ Exists fun k => And (GT.gt k 0) (Eq (Nat.card (Subtype fun x => Membership.m …
  -/
  have : Finite G := Nat.finite_of_card_ne_zero (hGpn ▸ pow_ne_zero n (NeZero.ne p))
  /-
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Fact (Nat.Prime p)
    n : Nat
    hGpn : Eq (Nat.card G) (HPow.hPow p n)
    hn : LT.lt 0 n
    this : Finite G
    ⊢ Exists fun k => And (GT.gt k 0) (Eq (Nat.card (Subtype fun x => Membership.m …
  -/
  have hcG := to_subgroup (of_card hGpn) (center G)
  /-
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Fact (Nat.Prime p)
    n : Nat
    hGpn : Eq (Nat.card G) (HPow.hPow p n)
    hn : LT.lt 0 n
    this : Finite G
    hcG : IsPGroup p (Subtype fun x => Membership.mem (Subgroup.center G) x)
    ⊢ Exists fun k => And (GT.gt k 0) (Eq (Nat.card (Subtype fun x => Membership.m …
  -/
  rcases iff_card.1 hcG with _
  /-
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Fact (Nat.Prime p)
    n : Nat
    hGpn : Eq (Nat.card G) (HPow.hPow p n)
    hn : LT.lt 0 n
    this : Finite G
    hcG : IsPGroup p (Subtype fun x => Membership.mem (Subgroup.center G) x)
    x✝ : Exists fun n => Eq (Nat.card (Subtype fun x => Membership.mem (Subgroup.c …
    ⊢ Exists fun k => And (GT.gt k 0) (Eq (Nat.card (Subtype fun x => Membership.m …
  -/
  haveI : Nontrivial G := (nontrivial_iff_card <| of_card hGpn).2 ⟨n, hn, hGpn⟩
  /-
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Fact (Nat.Prime p)
    n : Nat
    hGpn : Eq (Nat.card G) (HPow.hPow p n)
    hn : LT.lt 0 n
    this✝ : Finite G
    hcG : IsPGroup p (Subtype fun x => Membership.mem (Subgroup.center G) x)
    x✝ : Exists fun n => Eq (Nat.card (Subtype fun x => Membership.mem (Subgroup.c …
    this : Nontrivial G
    ⊢ Exists fun k => And (GT.gt k 0) (Eq (Nat.card (Subtype fun x => Membership.m …
  -/
  exact (nontrivial_iff_card hcG).mp (center_nontrivial (of_card hGpn))
  /-
    🎉 no goals
  -/


/-- The quotient by the center of a group of cardinality `p ^ 2` is cyclic. -/
theorem cyclic_center_quotient_of_card_eq_prime_sq (hG : Nat.card G = p ^ 2) :
    IsCyclic (G ⧸ center G) := by
  /-
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Fact (Nat.Prime p)
    hG : Eq (Nat.card G) (HPow.hPow p 2)
    ⊢ IsCyclic (HasQuotient.Quotient G (Subgroup.center G))
  -/
  apply isCyclic_of_card_dvd_prime (p := p)
  /-
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Fact (Nat.Prime p)
    hG : Eq (Nat.card G) (HPow.hPow p 2)
    ⊢ Dvd.dvd (Nat.card (HasQuotient.Quotient G (Subgroup.center G))) p
  -/
  rw [← mul_dvd_mul_iff_left (NeZero.ne p), ← sq, ← hG, ← (center G).card_mul_index]
  /-
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Fact (Nat.Prime p)
    hG : Eq (Nat.card G) (HPow.hPow p 2)
    ⊢ Dvd.dvd (HMul.hMul p (Nat.card (HasQuotient.Quotient G (Subgroup.center G))) …
  -/
  apply mul_dvd_mul_right
  /-
    case h
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Fact (Nat.Prime p)
    hG : Eq (Nat.card G) (HPow.hPow p 2)
    ⊢ Dvd.dvd p (Nat.card (Subtype fun x => Membership.mem (Subgroup.center G) x))
  -/
  rcases card_center_eq_prime_pow hG zero_lt_two with ⟨k, hk0, hk⟩
  /-
    case h.intro.intro
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Fact (Nat.Prime p)
    hG : Eq (Nat.card G) (HPow.hPow p 2)
    k : Nat
    hk0 : GT.gt k 0
    hk : Eq (Nat.card (Subtype fun x => Membership.mem (Subgroup.center G) x)) (HP …
    ⊢ Dvd.dvd p (Nat.card (Subtype fun x => Membership.mem (Subgroup.center G) x))
  -/
  rw [hk]
  /-
    case h.intro.intro
    p : Nat
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Fact (Nat.Prime p)
    hG : Eq (Nat.card G) (HPow.hPow p 2)
    k : Nat
    hk0 : GT.gt k 0
    hk : Eq (Nat.card (Subtype fun x => Membership.mem (Subgroup.center G) x)) (HP …
    ⊢ Dvd.dvd p (HPow.hPow p k)
  -/
  exact dvd_pow_self p hk0.ne'
  /-
    🎉 no goals
  -/


/-- A group of order `p ^ 2` is commutative. See also `IsPGroup.commutative_of_card_eq_prime_sq`
for just the proof that `∀ a b, a * b = b * a` -/
def commGroupOfCardEqPrimeSq (hG : Nat.card G = p ^ 2) : CommGroup G :=
  @commGroupOfCyclicCenterQuotient _ _ _ _ (cyclic_center_quotient_of_card_eq_prime_sq hG) _
    (QuotientGroup.ker_mk' (center G)).le


/-- A group of order `p ^ 2` is commutative. See also `IsPGroup.commGroupOfCardEqPrimeSq`
for the `CommGroup` instance. -/
theorem commutative_of_card_eq_prime_sq (hG : Nat.card G = p ^ 2) : ∀ a b : G, a * b = b * a :=
  (commGroupOfCardEqPrimeSq hG).mul_comm


lemma isPGroup_multiplicative : IsPGroup n (Multiplicative G) := by
  simpa [IsPGroup, Multiplicative.forall] using
    fun _ ↦ ⟨1, by simp [← ofAdd_nsmul, ZModModule.char_nsmul_eq_zero]⟩


