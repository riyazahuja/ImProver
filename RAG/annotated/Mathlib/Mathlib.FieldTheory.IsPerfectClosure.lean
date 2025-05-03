/-- Given a natural number `p`, the `p`-nilradical of a ring is defined to be the
nilradical if `p > 1` (`pNilradical_eq_nilradical`), and defined to be the zero ideal if `p ≤ 1`
(`pNilradical_eq_bot'`). Equivalently, it is the ideal consisting of elements `x` such that
`x ^ p ^ n = 0` for some `n` (`mem_pNilradical`). -/
def pNilradical (R : Type*) [CommSemiring R] (p : ℕ) : Ideal R := if 1 < p then nilradical R else ⊥


theorem pNilradical_le_nilradical {R : Type*} [CommSemiring R] {p : ℕ} :
    pNilradical R p ≤ nilradical R := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    p : Nat
    ⊢ LE.le (pNilradical R p) (nilradical R)
  -/
  by_cases hp : 1 < p
    /-
      case pos
      R : Type u_1
      inst✝ : CommSemiring R
      p : Nat
      hp : LT.lt 1 p
      ⊢ LE.le (pNilradical R p) (nilradical R)
    -/
  · rw [pNilradical, if_pos hp]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝ : CommSemiring R
    p : Nat
    hp : Not (LT.lt 1 p)
    ⊢ LE.le (pNilradical R p) (nilradical R)
  -/
  simp_rw [pNilradical, if_neg hp, bot_le]
  /-
    🎉 no goals
  -/


theorem pNilradical_eq_nilradical {R : Type*} [CommSemiring R] {p : ℕ} (hp : 1 < p) :
                                         /-
                                           R : Type u_1
                                           inst✝ : CommSemiring R
                                           p : Nat
                                           hp : LT.lt 1 p
                                           ⊢ Eq (pNilradical R p) (nilradical R)
                                         -/
    pNilradical R p = nilradical R := by rw [pNilradical, if_pos hp]
                                         /-
                                           🎉 no goals
                                         -/


theorem pNilradical_eq_bot {R : Type*} [CommSemiring R] {p : ℕ} (hp : ¬ 1 < p) :
                              /-
                                R : Type u_1
                                inst✝ : CommSemiring R
                                p : Nat
                                hp : Not (LT.lt 1 p)
                                ⊢ Eq (pNilradical R p) Bot.bot
                              -/
    pNilradical R p = ⊥ := by rw [pNilradical, if_neg hp]
                              /-
                                🎉 no goals
                              -/


theorem pNilradical_eq_bot' {R : Type*} [CommSemiring R] {p : ℕ} (hp : p ≤ 1) :
    pNilradical R p = ⊥ := pNilradical_eq_bot (not_lt.2 hp)


theorem pNilradical_prime {R : Type*} [CommSemiring R] {p : ℕ} (hp : p.Prime) :
    pNilradical R p = nilradical R := pNilradical_eq_nilradical hp.one_lt


theorem pNilradical_one {R : Type*} [CommSemiring R] :
    pNilradical R 1 = ⊥ := pNilradical_eq_bot' rfl.le


theorem mem_pNilradical {R : Type*} [CommSemiring R] {p : ℕ} {x : R} :
    x ∈ pNilradical R p ↔ ∃ n : ℕ, x ^ p ^ n = 0 := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    p : Nat
    x : R
    ⊢ Iff (Membership.mem (pNilradical R p) x) (Exists fun n => Eq (HPow.hPow x (H …
  -/
  by_cases hp : 1 < p
    /-
      case pos
      R : Type u_1
      inst✝ : CommSemiring R
      p : Nat
      x : R
      hp : LT.lt 1 p
      ⊢ Iff (Membership.mem (pNilradical R p) x) (Exists fun n => Eq (HPow.hPow x (H …
    -/
  · rw [pNilradical_eq_nilradical hp]
    /-
      case pos
      R : Type u_1
      inst✝ : CommSemiring R
      p : Nat
      x : R
      hp : LT.lt 1 p
      ⊢ Iff (Membership.mem (nilradical R) x) (Exists fun n => Eq (HPow.hPow x (HPow …
    -/
    refine ⟨fun ⟨n, h⟩ ↦ ⟨n, ?_⟩, fun ⟨n, h⟩ ↦ ⟨p ^ n, h⟩⟩
    /-
      case pos
      R : Type u_1
      inst✝ : CommSemiring R
      p : Nat
      x : R
      hp : LT.lt 1 p
      x✝ : Membership.mem (nilradical R) x
      n : Nat
      h : Membership.mem 0 (HPow.hPow x n)
      ⊢ Eq (HPow.hPow x (HPow.hPow p n)) 0
    -/
    rw [← Nat.sub_add_cancel ((n.lt_pow_self hp).le), pow_add, h, mul_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝ : CommSemiring R
    p : Nat
    x : R
    hp : Not (LT.lt 1 p)
    ⊢ Iff (Membership.mem (pNilradical R p) x) (Exists fun n => Eq (HPow.hPow x (H …
  -/
  rw [pNilradical_eq_bot hp, Ideal.mem_bot]
  /-
    case neg
    R : Type u_1
    inst✝ : CommSemiring R
    p : Nat
    x : R
    hp : Not (LT.lt 1 p)
    ⊢ Iff (Eq x 0) (Exists fun n => Eq (HPow.hPow x (HPow.hPow p n)) 0)
  -/
  refine ⟨fun h ↦ ⟨0, by rw [pow_zero, pow_one, h]⟩, fun ⟨n, h⟩ ↦ ?_⟩
  /-
    case neg
    R : Type u_1
    inst✝ : CommSemiring R
    p : Nat
    x : R
    hp : Not (LT.lt 1 p)
    x✝ : Exists fun n => Eq (HPow.hPow x (HPow.hPow p n)) 0
    n : Nat
    h : Eq (HPow.hPow x (HPow.hPow p n)) 0
    ⊢ Eq x 0
  -/
  rcases Nat.le_one_iff_eq_zero_or_eq_one.1 (not_lt.1 hp) with hp | hp
    /-
      case neg.inl
      R : Type u_1
      inst✝ : CommSemiring R
      p : Nat
      x : R
      hp✝ : Not (LT.lt 1 p)
      x✝ : Exists fun n => Eq (HPow.hPow x (HPow.hPow p n)) 0
      n : Nat
      h : Eq (HPow.hPow x (HPow.hPow p n)) 0
      hp : Eq p 0
      ⊢ Eq x 0
    -/
  · by_cases hn : n = 0
      /-
        case pos
        R : Type u_1
        inst✝ : CommSemiring R
        p : Nat
        x : R
        hp✝ : Not (LT.lt 1 p)
        x✝ : Exists fun n => Eq (HPow.hPow x (HPow.hPow p n)) 0
        n : Nat
        h : Eq (HPow.hPow x (HPow.hPow p n)) 0
        hp : Eq p 0
        hn : Eq n 0
        ⊢ Eq x 0
      -/
    · rwa [hn, pow_zero, pow_one] at h
      /-
        🎉 no goals
      -/
    /-
      case neg
      R : Type u_1
      inst✝ : CommSemiring R
      p : Nat
      x : R
      hp✝ : Not (LT.lt 1 p)
      x✝ : Exists fun n => Eq (HPow.hPow x (HPow.hPow p n)) 0
      n : Nat
      h : Eq (HPow.hPow x (HPow.hPow p n)) 0
      hp : Eq p 0
      hn : Not (Eq n 0)
      ⊢ Eq x 0
    -/
    rw [hp, zero_pow hn, pow_zero] at h
    /-
      case neg
      R : Type u_1
      inst✝ : CommSemiring R
      p : Nat
      x : R
      hp✝ : Not (LT.lt 1 p)
      x✝ : Exists fun n => Eq (HPow.hPow x (HPow.hPow p n)) 0
      n : Nat
      h : Eq 1 0
      hp : Eq p 0
      hn : Not (Eq n 0)
      ⊢ Eq x 0
    -/
    subsingleton [subsingleton_of_zero_eq_one h.symm]
    /-
      🎉 no goals
    -/
  /-
    case neg.inr
    R : Type u_1
    inst✝ : CommSemiring R
    p : Nat
    x : R
    hp✝ : Not (LT.lt 1 p)
    x✝ : Exists fun n => Eq (HPow.hPow x (HPow.hPow p n)) 0
    n : Nat
    h : Eq (HPow.hPow x (HPow.hPow p n)) 0
    hp : Eq p 1
    ⊢ Eq x 0
  -/
  rwa [hp, one_pow, pow_one] at h
  /-
    🎉 no goals
  -/


theorem sub_mem_pNilradical_iff_pow_expChar_pow_eq {R : Type*} [CommRing R] {p : ℕ} [ExpChar R p]
    {x y : R} : x - y ∈ pNilradical R p ↔ ∃ n : ℕ, x ^ p ^ n = y ^ p ^ n := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    p : Nat
    inst✝ : ExpChar R p
    x y : R
    ⊢ Iff (Membership.mem (pNilradical R p) (HSub.hSub x y)) (Exists fun n => Eq ( …
  -/
  simp_rw [mem_pNilradical, sub_pow_expChar_pow, sub_eq_zero]
  /-
    🎉 no goals
  -/


theorem pow_expChar_pow_inj_of_pNilradical_eq_bot (R : Type*) [CommRing R] (p : ℕ) [ExpChar R p]
    (h : pNilradical R p = ⊥) (n : ℕ) : Function.Injective fun x : R ↦ x ^ p ^ n := fun _ _ H ↦
  sub_eq_zero.1 <| Ideal.mem_bot.1 <| h ▸ sub_mem_pNilradical_iff_pow_expChar_pow_eq.2 ⟨n, H⟩


theorem pNilradical_eq_bot_of_frobenius_inj (R : Type*) [CommRing R] (p : ℕ) [ExpChar R p]
    (h : Function.Injective (frobenius R p)) : pNilradical R p = ⊥ := bot_unique fun x ↦ by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    p : Nat
    inst✝ : ExpChar R p
    h : Function.Injective ⇑(frobenius R p)
    x : R
    ⊢ Membership.mem (pNilradical R p) x → Membership.mem Bot.bot x
  -/
  rw [mem_pNilradical, Ideal.mem_bot]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    p : Nat
    inst✝ : ExpChar R p
    h : Function.Injective ⇑(frobenius R p)
    x : R
    ⊢ (Exists fun n => Eq (HPow.hPow x (HPow.hPow p n)) 0) → Eq x 0
  -/
  exact fun ⟨n, _⟩ ↦ h.iterate n (by rwa [← coe_iterateFrobenius, map_zero])
  /-
    🎉 no goals
  -/


theorem PerfectRing.pNilradical_eq_bot (R : Type*) [CommRing R] (p : ℕ) [ExpChar R p]
    [PerfectRing R p] : pNilradical R p = ⊥ :=
  pNilradical_eq_bot_of_frobenius_inj R p (injective_frobenius R p)


/-- If `i : K →+* L` is a ring homomorphism of characteristic `p` rings, then it is called
`p`-radical if the following conditions are satisfied:

- For any element `x` of `L` there is `n : ℕ` such that `x ^ (p ^ n)` is contained in `K`.
- The kernel of `i` is contained in the `p`-nilradical of `K`.

It is a generalization of purely inseparable extension for fields. -/
@[mk_iff]
class IsPRadical : Prop where
  pow_mem' : ∀ x : L, ∃ (n : ℕ) (y : K), i y = x ^ p ^ n
  ker_le' : RingHom.ker i ≤ pNilradical K p


theorem IsPRadical.pow_mem [IsPRadical i p] (x : L) :
    ∃ (n : ℕ) (y : K), i y = x ^ p ^ n := pow_mem' x


theorem IsPRadical.ker_le [IsPRadical i p] :
    RingHom.ker i ≤ pNilradical K p := ker_le'


theorem IsPRadical.comap_pNilradical [IsPRadical i p] :
    (pNilradical L p).comap i = pNilradical K p := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝² : CommSemiring K
    inst✝¹ : CommSemiring L
    i : RingHom K L
    p : Nat
    inst✝ : IsPRadical i p
    ⊢ Eq (Ideal.comap i (pNilradical L p)) (pNilradical K p)
  -/
  refine le_antisymm (fun x h ↦ mem_pNilradical.2 ?_) (fun x h ↦ ?_)
    /-
      case refine_1
      K : Type u_1
      L : Type u_2
      inst✝² : CommSemiring K
      inst✝¹ : CommSemiring L
      i : RingHom K L
      p : Nat
      inst✝ : IsPRadical i p
      x : K
      h : Membership.mem (Ideal.comap i (pNilradical L p)) x
      ⊢ Exists fun n => Eq (HPow.hPow x (HPow.hPow p n)) 0
    -/
  · obtain ⟨n, h⟩ := mem_pNilradical.1 <| Ideal.mem_comap.1 h
    /-
      case refine_1.intro
      K : Type u_1
      L : Type u_2
      inst✝² : CommSemiring K
      inst✝¹ : CommSemiring L
      i : RingHom K L
      p : Nat
      inst✝ : IsPRadical i p
      x : K
      h✝ : Membership.mem (Ideal.comap i (pNilradical L p)) x
      n : Nat
      h : Eq (HPow.hPow (i x) (HPow.hPow p n)) 0
      ⊢ Exists fun n => Eq (HPow.hPow x (HPow.hPow p n)) 0
    -/
    obtain ⟨m, h⟩ := mem_pNilradical.1 <| ker_le i p ((map_pow i x _).symm ▸ h)
    /-
      case refine_1.intro.intro
      K : Type u_1
      L : Type u_2
      inst✝² : CommSemiring K
      inst✝¹ : CommSemiring L
      i : RingHom K L
      p : Nat
      inst✝ : IsPRadical i p
      x : K
      h✝¹ : Membership.mem (Ideal.comap i (pNilradical L p)) x
      n : Nat
      h✝ : Eq (HPow.hPow (i x) (HPow.hPow p n)) 0
      m : Nat
      h : Eq (HPow.hPow (HPow.hPow x (HPow.hPow p n)) (HPow.hPow p m)) 0
      ⊢ Exists fun n => Eq (HPow.hPow x (HPow.hPow p n)) 0
    -/
    exact ⟨n + m, by rwa [pow_add, pow_mul]⟩
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    K : Type u_1
    L : Type u_2
    inst✝² : CommSemiring K
    inst✝¹ : CommSemiring L
    i : RingHom K L
    p : Nat
    inst✝ : IsPRadical i p
    x : K
    h : Membership.mem (pNilradical K p) x
    ⊢ Membership.mem (Ideal.comap i (pNilradical L p)) x
  -/
  simp only [Ideal.mem_comap, mem_pNilradical] at h ⊢
  /-
    case refine_2
    K : Type u_1
    L : Type u_2
    inst✝² : CommSemiring K
    inst✝¹ : CommSemiring L
    i : RingHom K L
    p : Nat
    inst✝ : IsPRadical i p
    x : K
    h : Exists fun n => Eq (HPow.hPow x (HPow.hPow p n)) 0
    ⊢ Exists fun n => Eq (HPow.hPow (i x) (HPow.hPow p n)) 0
  -/
  obtain ⟨n, h⟩ := h
  /-
    case refine_2.intro
    K : Type u_1
    L : Type u_2
    inst✝² : CommSemiring K
    inst✝¹ : CommSemiring L
    i : RingHom K L
    p : Nat
    inst✝ : IsPRadical i p
    x : K
    n : Nat
    h : Eq (HPow.hPow x (HPow.hPow p n)) 0
    ⊢ Exists fun n => Eq (HPow.hPow (i x) (HPow.hPow p n)) 0
  -/
  exact ⟨n, by simpa only [map_pow, map_zero] using congr(i $h)⟩
  /-
    🎉 no goals
  -/


variable (K) in
instance IsPRadical.of_id : IsPRadical (RingHom.id K) p where
                          /-
                            K : Type u_1
                            L : Type u_2
                            M : Type u_3
                            N : Type u_4
                            inst✝² : CommSemiring K
                            inst✝¹ : CommSemiring L
                            inst✝ : CommSemiring M
                            i : RingHom K L
                            j : RingHom K M
                            f : RingHom L M
                            p : Nat
                            x : K
                            ⊢ Eq ((RingHom.id K) x) (HPow.hPow x (HPow.hPow p 0))
                          -/
  pow_mem' x := ⟨0, x, by simp⟩
                          /-
                            🎉 no goals
                          -/
                    /-
                      K : Type u_1
                      L : Type u_2
                      M : Type u_3
                      N : Type u_4
                      inst✝² : CommSemiring K
                      inst✝¹ : CommSemiring L
                      inst✝ : CommSemiring M
                      i : RingHom K L
                      j : RingHom K M
                      f : RingHom L M
                      p : Nat
                      x : K
                      h : Membership.mem (RingHom.ker (RingHom.id K)) x
                      ⊢ Membership.mem (pNilradical K p) x
                    -/
  ker_le' x h := by convert Ideal.zero_mem _
                    /-
                      🎉 no goals
                    -/


/-- Composition of `p`-radical ring homomorphisms is also `p`-radical. -/
theorem IsPRadical.trans [IsPRadical i p] [IsPRadical f p] :
    IsPRadical (f.comp i) p where
  pow_mem' x := by
    /-
      K : Type u_1
      L : Type u_2
      M : Type u_3
      inst✝⁴ : CommSemiring K
      inst✝³ : CommSemiring L
      inst✝² : CommSemiring M
      i : RingHom K L
      f : RingHom L M
      p : Nat
      inst✝¹ : IsPRadical i p
      inst✝ : IsPRadical f p
      x : M
      ⊢ Exists fun n => Exists fun y => Eq ((f.comp i) y) (HPow.hPow x (HPow.hPow p  …
    -/
    obtain ⟨n, y, hy⟩ := pow_mem f p x
    /-
      case intro.intro
      K : Type u_1
      L : Type u_2
      M : Type u_3
      inst✝⁴ : CommSemiring K
      inst✝³ : CommSemiring L
      inst✝² : CommSemiring M
      i : RingHom K L
      f : RingHom L M
      p : Nat
      inst✝¹ : IsPRadical i p
      inst✝ : IsPRadical f p
      x : M
      n : Nat
      y : L
      hy : Eq (f y) (HPow.hPow x (HPow.hPow p n))
      ⊢ Exists fun n => Exists fun y => Eq ((f.comp i) y) (HPow.hPow x (HPow.hPow p  …
    -/
    obtain ⟨m, z, hz⟩ := pow_mem i p y
    /-
      case intro.intro.intro.intro
      K : Type u_1
      L : Type u_2
      M : Type u_3
      inst✝⁴ : CommSemiring K
      inst✝³ : CommSemiring L
      inst✝² : CommSemiring M
      i : RingHom K L
      f : RingHom L M
      p : Nat
      inst✝¹ : IsPRadical i p
      inst✝ : IsPRadical f p
      x : M
      n : Nat
      y : L
      hy : Eq (f y) (HPow.hPow x (HPow.hPow p n))
      m : Nat
      z : K
      hz : Eq (i z) (HPow.hPow y (HPow.hPow p m))
      ⊢ Exists fun n => Exists fun y => Eq ((f.comp i) y) (HPow.hPow x (HPow.hPow p  …
    -/
    exact ⟨n + m, z, by rw [RingHom.comp_apply, hz, map_pow, hy, pow_add, pow_mul]⟩
    /-
      🎉 no goals
    -/
  ker_le' x h := by
    /-
      K : Type u_1
      L : Type u_2
      M : Type u_3
      inst✝⁴ : CommSemiring K
      inst✝³ : CommSemiring L
      inst✝² : CommSemiring M
      i : RingHom K L
      f : RingHom L M
      p : Nat
      inst✝¹ : IsPRadical i p
      inst✝ : IsPRadical f p
      x : K
      h : Membership.mem (RingHom.ker (f.comp i)) x
      ⊢ Membership.mem (pNilradical K p) x
    -/
    rw [RingHom.mem_ker, RingHom.comp_apply, ← RingHom.mem_ker] at h
    /-
      K : Type u_1
      L : Type u_2
      M : Type u_3
      inst✝⁴ : CommSemiring K
      inst✝³ : CommSemiring L
      inst✝² : CommSemiring M
      i : RingHom K L
      f : RingHom L M
      p : Nat
      inst✝¹ : IsPRadical i p
      inst✝ : IsPRadical f p
      x : K
      h : Membership.mem (RingHom.ker f) (i x)
      ⊢ Membership.mem (pNilradical K p) x
    -/
    simpa only [← Ideal.mem_comap, comap_pNilradical] using ker_le f p h
    /-
      🎉 no goals
    -/


/-- If `i : K →+* L` is a `p`-radical ring homomorphism, then it makes `L` a perfect closure
of `K`, if `L` is perfect.
In this case the kernel of `i` is equal to the `p`-nilradical of `K`
(see `IsPerfectClosure.ker_eq`).

Our definition makes it synonymous to `IsPRadical` if `PerfectRing L p` is present. A caveat is
that you need to write `[PerfectRing L p] [IsPerfectClosure i p]`. This is similar to
`PerfectRing` which has `ExpChar` as a prerequisite. -/
@[nolint unusedArguments]
abbrev IsPerfectClosure [ExpChar L p] [PerfectRing L p] := IsPRadical i p


/-- If `i : K →+* L` is a ring homomorphism of exponential characteristic `p` rings, such that `L`
is perfect, then the `p`-nilradical of `K` is contained in the kernel of `i`. -/
theorem RingHom.pNilradical_le_ker_of_perfectRing [ExpChar L p] [PerfectRing L p] :
    pNilradical K p ≤ RingHom.ker i := fun x h ↦ by
  /-
    K : Type u_1
    L : Type u_2
    inst✝³ : CommSemiring K
    inst✝² : CommSemiring L
    i : RingHom K L
    p : Nat
    inst✝¹ : ExpChar L p
    inst✝ : PerfectRing L p
    x : K
    h : Membership.mem (pNilradical K p) x
    ⊢ Membership.mem (RingHom.ker i) x
  -/
  obtain ⟨n, h⟩ := mem_pNilradical.1 h
  /-
    case intro
    K : Type u_1
    L : Type u_2
    inst✝³ : CommSemiring K
    inst✝² : CommSemiring L
    i : RingHom K L
    p : Nat
    inst✝¹ : ExpChar L p
    inst✝ : PerfectRing L p
    x : K
    h✝ : Membership.mem (pNilradical K p) x
    n : Nat
    h : Eq (HPow.hPow x (HPow.hPow p n)) 0
    ⊢ Membership.mem (RingHom.ker i) x
  -/
  replace h := congr((iterateFrobeniusEquiv L p n).symm (i $h))
  rwa [map_pow, ← iterateFrobenius_def, ← iterateFrobeniusEquiv_apply, RingEquiv.symm_apply_apply,
    map_zero, map_zero] at h


variable [ExpChar L p] in
theorem IsPerfectClosure.ker_eq [PerfectRing L p] [IsPerfectClosure i p] :
    RingHom.ker i = pNilradical K p :=
  IsPRadical.ker_le'.antisymm (i.pNilradical_le_ker_of_perfectRing p)


theorem lift_aux (x : L) : ∃ y : ℕ × K, i y.2 = x ^ p ^ y.1 := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝² : CommSemiring K
    inst✝¹ : CommSemiring L
    i : RingHom K L
    p : Nat
    inst✝ : IsPRadical i p
    x : L
    ⊢ Exists fun y => Eq (i y.2) (HPow.hPow x (HPow.hPow p y.1))
  -/
  obtain ⟨n, y, h⟩ := IsPRadical.pow_mem i p x
  /-
    case intro.intro
    K : Type u_1
    L : Type u_2
    inst✝² : CommSemiring K
    inst✝¹ : CommSemiring L
    i : RingHom K L
    p : Nat
    inst✝ : IsPRadical i p
    x : L
    n : Nat
    y : K
    h : Eq (i y) (HPow.hPow x (HPow.hPow p n))
    ⊢ Exists fun y => Eq (i y.2) (HPow.hPow x (HPow.hPow p y.1))
  -/
  exact ⟨(n, y), h⟩
  /-
    🎉 no goals
  -/


/-- If `i : K →+* L` and `j : K →+* M` are ring homomorphisms of characteristic `p` rings, such that
`i` is `p`-radical (in fact only the `IsPRadical.pow_mem` is required) and `M` is a perfect ring,
then one can define a map `L → M` which maps an element `x` of `L` to `y ^ (p ^ -n)` if
`x ^ (p ^ n)` is equal to some element `y` of `K`. -/
def liftAux (x : L) : M := (iterateFrobeniusEquiv M p (Classical.choose (lift_aux i p x)).1).symm
  (j (Classical.choose (lift_aux i p x)).2)


@[simp]
theorem liftAux_self_apply [ExpChar L p] [PerfectRing L p] (x : L) : liftAux i i p x = x := by
  rw [liftAux, Classical.choose_spec (lift_aux i p x), ← iterateFrobenius_def,
    ← iterateFrobeniusEquiv_apply, RingEquiv.symm_apply_apply]


@[simp]
theorem liftAux_self [ExpChar L p] [PerfectRing L p] : liftAux i i p = id :=
  funext (liftAux_self_apply i p)


@[simp]
theorem liftAux_id_apply (x : K) : liftAux (RingHom.id K) j p x = j x := by
  /-
    K : Type u_1
    M : Type u_3
    inst✝³ : CommSemiring K
    inst✝² : CommSemiring M
    j : RingHom K M
    p : Nat
    inst✝¹ : ExpChar M p
    inst✝ : PerfectRing M p
    x : K
    ⊢ Eq (PerfectRing.liftAux (RingHom.id K) j p x) (j x)
  -/
  have := RingHom.id_apply _ ▸ Classical.choose_spec (lift_aux (RingHom.id K) p x)
  rw [liftAux, this, map_pow, ← iterateFrobenius_def, ← iterateFrobeniusEquiv_apply,
    RingEquiv.symm_apply_apply]


@[simp]
theorem liftAux_id : liftAux (RingHom.id K) j p = j := funext (liftAux_id_apply j p)


/-- If `i : K →+* L` is `p`-radical, then for any ring `M` of exponential charactistic `p` whose
`p`-nilradical is zero, the map `(L →+* M) → (K →+* M)` induced by `i` is injective. -/
theorem injective_comp_of_pNilradical_eq_bot [IsPRadical i p] (h : pNilradical M p = ⊥) :
    Function.Injective fun f : L →+* M ↦ f.comp i := fun f g heq ↦ by
  /-
    K : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁴ : CommRing K
    inst✝³ : CommRing L
    inst✝² : CommRing M
    i : RingHom K L
    p : Nat
    inst✝¹ : ExpChar M p
    inst✝ : IsPRadical i p
    h : Eq (pNilradical M p) Bot.bot
    f g : RingHom L M
    heq : Eq ((fun f => f.comp i) f) ((fun f => f.comp i) g)
    ⊢ Eq f g
  -/
  ext x
  /-
    case a
    K : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁴ : CommRing K
    inst✝³ : CommRing L
    inst✝² : CommRing M
    i : RingHom K L
    p : Nat
    inst✝¹ : ExpChar M p
    inst✝ : IsPRadical i p
    h : Eq (pNilradical M p) Bot.bot
    f g : RingHom L M
    heq : Eq ((fun f => f.comp i) f) ((fun f => f.comp i) g)
    x : L
    ⊢ Eq (f x) (g x)
  -/
  obtain ⟨n, y, hx⟩ := IsPRadical.pow_mem i p x
  /-
    case a.intro.intro
    K : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁴ : CommRing K
    inst✝³ : CommRing L
    inst✝² : CommRing M
    i : RingHom K L
    p : Nat
    inst✝¹ : ExpChar M p
    inst✝ : IsPRadical i p
    h : Eq (pNilradical M p) Bot.bot
    f g : RingHom L M
    heq : Eq ((fun f => f.comp i) f) ((fun f => f.comp i) g)
    x : L
    n : Nat
    y : K
    hx : Eq (i y) (HPow.hPow x (HPow.hPow p n))
    ⊢ Eq (f x) (g x)
  -/
  apply_fun _ using pow_expChar_pow_inj_of_pNilradical_eq_bot M p h n
  /-
    case a.intro.intro
    K : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁴ : CommRing K
    inst✝³ : CommRing L
    inst✝² : CommRing M
    i : RingHom K L
    p : Nat
    inst✝¹ : ExpChar M p
    inst✝ : IsPRadical i p
    h : Eq (pNilradical M p) Bot.bot
    f g : RingHom L M
    heq : Eq ((fun f => f.comp i) f) ((fun f => f.comp i) g)
    x : L
    n : Nat
    y : K
    hx : Eq (i y) (HPow.hPow x (HPow.hPow p n))
    ⊢ Eq (HPow.hPow (f x) (HPow.hPow p n)) (HPow.hPow (g x) (HPow.hPow p n))
  -/
  simpa only [← map_pow, ← hx] using congr($(heq) y)
  /-
    🎉 no goals
  -/


/-- If `i : K →+* L` is `p`-radical, then for any reduced ring `M` of exponential charactistic `p`,
the map `(L →+* M) → (K →+* M)` induced by `i` is injective.
A special case of `IsPRadical.injective_comp_of_pNilradical_eq_bot`
and a generalization of `IsPurelyInseparable.injective_comp_algebraMap`. -/
theorem injective_comp [IsPRadical i p] [IsReduced M] :
    Function.Injective fun f : L →+* M ↦ f.comp i :=
  injective_comp_of_pNilradical_eq_bot i p <| bot_unique <|
    pNilradical_le_nilradical.trans (nilradical_eq_zero M).le


/-- If `i : K →+* L` is `p`-radical, then for any perfect ring `M` of exponential charactistic `p`,
the map `(L →+* M) → (K →+* M)` induced by `i` is injective.
A special case of `IsPRadical.injective_comp_of_pNilradical_eq_bot`. -/
theorem injective_comp_of_perfect [IsPRadical i p] [PerfectRing M p] :
    Function.Injective fun f : L →+* M ↦ f.comp i :=
  injective_comp_of_pNilradical_eq_bot i p (PerfectRing.pNilradical_eq_bot M p)


/-- If `i : K →+* L` and `j : K →+* M` are ring homomorphisms of characteristic `p` rings, such that
`i` is `p`-radical, and `M` is a perfect ring, then `PerfectRing.liftAux` is well-defined. -/
theorem liftAux_apply (x : L) (n : ℕ) (y : K) (h : i y = x ^ p ^ n) :
    liftAux i j p x = (iterateFrobeniusEquiv M p n).symm (j y) := by
  /-
    K : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁶ : CommRing K
    inst✝⁵ : CommRing L
    inst✝⁴ : CommRing M
    i : RingHom K L
    j : RingHom K M
    p : Nat
    inst✝³ : ExpChar M p
    inst✝² : ExpChar K p
    inst✝¹ : PerfectRing M p
    inst✝ : IsPRadical i p
    x : L
    n : Nat
    y : K
    h : Eq (i y) (HPow.hPow x (HPow.hPow p n))
    ⊢ Eq (PerfectRing.liftAux i j p x) ((iterateFrobeniusEquiv M p n).symm (j y))
  -/
  rw [liftAux]
  /-
    K : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁶ : CommRing K
    inst✝⁵ : CommRing L
    inst✝⁴ : CommRing M
    i : RingHom K L
    j : RingHom K M
    p : Nat
    inst✝³ : ExpChar M p
    inst✝² : ExpChar K p
    inst✝¹ : PerfectRing M p
    inst✝ : IsPRadical i p
    x : L
    n : Nat
    y : K
    h : Eq (i y) (HPow.hPow x (HPow.hPow p n))
    ⊢ Eq ((iterateFrobeniusEquiv M p (Classical.choose ⋯).1).symm (j (Classical.ch …
  -/
  have h' := Classical.choose_spec (lift_aux i p x)
  /-
    K : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁶ : CommRing K
    inst✝⁵ : CommRing L
    inst✝⁴ : CommRing M
    i : RingHom K L
    j : RingHom K M
    p : Nat
    inst✝³ : ExpChar M p
    inst✝² : ExpChar K p
    inst✝¹ : PerfectRing M p
    inst✝ : IsPRadical i p
    x : L
    n : Nat
    y : K
    h : Eq (i y) (HPow.hPow x (HPow.hPow p n))
    h' : Eq (i (Classical.choose ⋯).2) (HPow.hPow x (HPow.hPow p (Classical.choose …
    ⊢ Eq ((iterateFrobeniusEquiv M p (Classical.choose ⋯).1).symm (j (Classical.ch …
  -/
  set n' := (Classical.choose (lift_aux i p x)).1
  /-
    K : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁶ : CommRing K
    inst✝⁵ : CommRing L
    inst✝⁴ : CommRing M
    i : RingHom K L
    j : RingHom K M
    p : Nat
    inst✝³ : ExpChar M p
    inst✝² : ExpChar K p
    inst✝¹ : PerfectRing M p
    inst✝ : IsPRadical i p
    x : L
    n : Nat
    y : K
    h : Eq (i y) (HPow.hPow x (HPow.hPow p n))
    n' : Nat := (Classical.choose ⋯).1
    h' : Eq (i (Classical.choose ⋯).2) (HPow.hPow x (HPow.hPow p n'))
    ⊢ Eq ((iterateFrobeniusEquiv M p n').symm (j (Classical.choose ⋯).2)) ((iterat …
  -/
  replace h := congr($(h.symm) ^ p ^ n')
  rw [← pow_mul, mul_comm, pow_mul, ← h', ← map_pow, ← map_pow, ← sub_eq_zero, ← map_sub,
    ← RingHom.mem_ker] at h
  /-
    K : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁶ : CommRing K
    inst✝⁵ : CommRing L
    inst✝⁴ : CommRing M
    i : RingHom K L
    j : RingHom K M
    p : Nat
    inst✝³ : ExpChar M p
    inst✝² : ExpChar K p
    inst✝¹ : PerfectRing M p
    inst✝ : IsPRadical i p
    x : L
    n : Nat
    y : K
    n' : Nat := (Classical.choose ⋯).1
    h' : Eq (i (Classical.choose ⋯).2) (HPow.hPow x (HPow.hPow p n'))
    h : Membership.mem (RingHom.ker i) (HSub.hSub (HPow.hPow (Classical.choose ⋯). …
    ⊢ Eq ((iterateFrobeniusEquiv M p n').symm (j (Classical.choose ⋯).2)) ((iterat …
  -/
  obtain ⟨m, h⟩ := mem_pNilradical.1 (IsPRadical.ker_le i p h)
  /-
    case intro
    K : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁶ : CommRing K
    inst✝⁵ : CommRing L
    inst✝⁴ : CommRing M
    i : RingHom K L
    j : RingHom K M
    p : Nat
    inst✝³ : ExpChar M p
    inst✝² : ExpChar K p
    inst✝¹ : PerfectRing M p
    inst✝ : IsPRadical i p
    x : L
    n : Nat
    y : K
    n' : Nat := (Classical.choose ⋯).1
    h' : Eq (i (Classical.choose ⋯).2) (HPow.hPow x (HPow.hPow p n'))
    h✝ : Membership.mem (RingHom.ker i) (HSub.hSub (HPow.hPow (Classical.choose ⋯) …
    m : Nat
    h : Eq (HPow.hPow (HSub.hSub (HPow.hPow (Classical.choose ⋯).2 (HPow.hPow p n) …
    ⊢ Eq ((iterateFrobeniusEquiv M p n').symm (j (Classical.choose ⋯).2)) ((iterat …
  -/
  refine (iterateFrobeniusEquiv M p (m + n + n')).injective ?_
  /-
    case intro
    K : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁶ : CommRing K
    inst✝⁵ : CommRing L
    inst✝⁴ : CommRing M
    i : RingHom K L
    j : RingHom K M
    p : Nat
    inst✝³ : ExpChar M p
    inst✝² : ExpChar K p
    inst✝¹ : PerfectRing M p
    inst✝ : IsPRadical i p
    x : L
    n : Nat
    y : K
    n' : Nat := (Classical.choose ⋯).1
    h' : Eq (i (Classical.choose ⋯).2) (HPow.hPow x (HPow.hPow p n'))
    h✝ : Membership.mem (RingHom.ker i) (HSub.hSub (HPow.hPow (Classical.choose ⋯) …
    m : Nat
    h : Eq (HPow.hPow (HSub.hSub (HPow.hPow (Classical.choose ⋯).2 (HPow.hPow p n) …
    ⊢ Eq ((iterateFrobeniusEquiv M p (HAdd.hAdd (HAdd.hAdd m n) n')) ((iterateFrob …
  -/
  conv_lhs => rw [iterateFrobeniusEquiv_add_apply, RingEquiv.apply_symm_apply]
  rw [add_assoc, add_comm n n', ← add_assoc,
    iterateFrobeniusEquiv_add_apply (m := m + n'), RingEquiv.apply_symm_apply,
    iterateFrobeniusEquiv_def, iterateFrobeniusEquiv_def,
    ← sub_eq_zero, ← map_pow, ← map_pow, ← map_sub,
    add_comm m, add_comm m, pow_add, pow_mul, pow_add, pow_mul, ← sub_pow_expChar_pow, h, map_zero]


/-- If `i : K →+* L` and `j : K →+* M` are ring homomorphisms of characteristic `p` rings, such that
`i` is `p`-radical, and `M` is a perfect ring, then `PerfectRing.liftAux`
is a ring homomorphism. This is similar to `IsAlgClosed.lift` and `IsSepClosed.lift`. -/
def lift : L →+* M where
  toFun := liftAux i j p
                 /-
                   K : Type u_1
                   L : Type u_2
                   M : Type u_3
                   N : Type u_4
                   inst✝⁸ : CommRing K
                   inst✝⁷ : CommRing L
                   inst✝⁶ : CommRing M
                   inst✝⁵ : CommRing N
                   i : RingHom K L
                   j : RingHom K M
                   k : RingHom K N
                   f : RingHom L M
                   g : RingHom L N
                   p : Nat
                   inst✝⁴ : ExpChar M p
                   inst✝³ : ExpChar K p
                   inst✝² : PerfectRing M p
                   inst✝¹ : IsPRadical i p
                   inst✝ : ExpChar L p
                   ⊢ Eq (PerfectRing.liftAux i j p 1) 1
                 -/
  map_one' := by simp [liftAux_apply i j p 1 0 1 (by rw [one_pow, map_one])]
                 /-
                   🎉 no goals
                 -/
  map_mul' x1 x2 := by
    /-
      K : Type u_1
      L : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁸ : CommRing K
      inst✝⁷ : CommRing L
      inst✝⁶ : CommRing M
      inst✝⁵ : CommRing N
      i : RingHom K L
      j : RingHom K M
      k : RingHom K N
      f : RingHom L M
      g : RingHom L N
      p : Nat
      inst✝⁴ : ExpChar M p
      inst✝³ : ExpChar K p
      inst✝² : PerfectRing M p
      inst✝¹ : IsPRadical i p
      inst✝ : ExpChar L p
      x1 x2 : L
      ⊢ Eq ({ toFun := PerfectRing.liftAux i j p, map_one' := ⋯ }.toFun (HMul.hMul x …
    -/
    obtain ⟨n1, y1, h1⟩ := IsPRadical.pow_mem i p x1
    /-
      case intro.intro
      K : Type u_1
      L : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁸ : CommRing K
      inst✝⁷ : CommRing L
      inst✝⁶ : CommRing M
      inst✝⁵ : CommRing N
      i : RingHom K L
      j : RingHom K M
      k : RingHom K N
      f : RingHom L M
      g : RingHom L N
      p : Nat
      inst✝⁴ : ExpChar M p
      inst✝³ : ExpChar K p
      inst✝² : PerfectRing M p
      inst✝¹ : IsPRadical i p
      inst✝ : ExpChar L p
      x1 x2 : L
      n1 : Nat
      y1 : K
      h1 : Eq (i y1) (HPow.hPow x1 (HPow.hPow p n1))
      ⊢ Eq ({ toFun := PerfectRing.liftAux i j p, map_one' := ⋯ }.toFun (HMul.hMul x …
    -/
    obtain ⟨n2, y2, h2⟩ := IsPRadical.pow_mem i p x2
    /-
      case intro.intro.intro.intro
      K : Type u_1
      L : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁸ : CommRing K
      inst✝⁷ : CommRing L
      inst✝⁶ : CommRing M
      inst✝⁵ : CommRing N
      i : RingHom K L
      j : RingHom K M
      k : RingHom K N
      f : RingHom L M
      g : RingHom L N
      p : Nat
      inst✝⁴ : ExpChar M p
      inst✝³ : ExpChar K p
      inst✝² : PerfectRing M p
      inst✝¹ : IsPRadical i p
      inst✝ : ExpChar L p
      x1 x2 : L
      n1 : Nat
      y1 : K
      h1 : Eq (i y1) (HPow.hPow x1 (HPow.hPow p n1))
      n2 : Nat
      y2 : K
      h2 : Eq (i y2) (HPow.hPow x2 (HPow.hPow p n2))
      ⊢ Eq ({ toFun := PerfectRing.liftAux i j p, map_one' := ⋯ }.toFun (HMul.hMul x …
    -/
    simp only; rw [liftAux_apply i j p _ _ _ h1, liftAux_apply i j p _ _ _ h2,
      liftAux_apply i j p (x1 * x2) (n1 + n2) (y1 ^ p ^ n2 * y2 ^ p ^ n1) (by rw [map_mul,
        map_pow, map_pow, h1, h2, ← pow_mul, ← pow_add, ← pow_mul, ← pow_add,
        add_comm n2, mul_pow]),
      map_mul, map_pow, map_pow, map_mul, ← iterateFrobeniusEquiv_def]
    /-
      case intro.intro.intro.intro
      K : Type u_1
      L : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁸ : CommRing K
      inst✝⁷ : CommRing L
      inst✝⁶ : CommRing M
      inst✝⁵ : CommRing N
      i : RingHom K L
      j : RingHom K M
      k : RingHom K N
      f : RingHom L M
      g : RingHom L N
      p : Nat
      inst✝⁴ : ExpChar M p
      inst✝³ : ExpChar K p
      inst✝² : PerfectRing M p
      inst✝¹ : IsPRadical i p
      inst✝ : ExpChar L p
      x1 x2 : L
      n1 : Nat
      y1 : K
      h1 : Eq (i y1) (HPow.hPow x1 (HPow.hPow p n1))
      n2 : Nat
      y2 : K
      h2 : Eq (i y2) (HPow.hPow x2 (HPow.hPow p n2))
      ⊢ Eq (HMul.hMul ((iterateFrobeniusEquiv M p (HAdd.hAdd n1 n2)).symm ((iterateF …
    -/
    nth_rw 1 [iterateFrobeniusEquiv_symm_add_apply]
    rw [RingEquiv.symm_apply_apply, add_comm n1, iterateFrobeniusEquiv_symm_add_apply,
      ← iterateFrobeniusEquiv_def, RingEquiv.symm_apply_apply]
                  /-
                    K : Type u_1
                    L : Type u_2
                    M : Type u_3
                    N : Type u_4
                    inst✝⁸ : CommRing K
                    inst✝⁷ : CommRing L
                    inst✝⁶ : CommRing M
                    inst✝⁵ : CommRing N
                    i : RingHom K L
                    j : RingHom K M
                    k : RingHom K N
                    f : RingHom L M
                    g : RingHom L N
                    p : Nat
                    inst✝⁴ : ExpChar M p
                    inst✝³ : ExpChar K p
                    inst✝² : PerfectRing M p
                    inst✝¹ : IsPRadical i p
                    inst✝ : ExpChar L p
                    ⊢ Eq ((↑{ toFun := PerfectRing.liftAux i j p, map_one' := ⋯, map_mul' := ⋯ }). …
                  -/
  map_zero' := by simp [liftAux_apply i j p 0 0 0 (by rw [pow_zero, pow_one, map_zero])]
                  /-
                    🎉 no goals
                  -/
  map_add' x1 x2 := by
    /-
      K : Type u_1
      L : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁸ : CommRing K
      inst✝⁷ : CommRing L
      inst✝⁶ : CommRing M
      inst✝⁵ : CommRing N
      i : RingHom K L
      j : RingHom K M
      k : RingHom K N
      f : RingHom L M
      g : RingHom L N
      p : Nat
      inst✝⁴ : ExpChar M p
      inst✝³ : ExpChar K p
      inst✝² : PerfectRing M p
      inst✝¹ : IsPRadical i p
      inst✝ : ExpChar L p
      x1 x2 : L
      ⊢ Eq ((↑{ toFun := PerfectRing.liftAux i j p, map_one' := ⋯, map_mul' := ⋯ }). …
    -/
    obtain ⟨n1, y1, h1⟩ := IsPRadical.pow_mem i p x1
    /-
      case intro.intro
      K : Type u_1
      L : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁸ : CommRing K
      inst✝⁷ : CommRing L
      inst✝⁶ : CommRing M
      inst✝⁵ : CommRing N
      i : RingHom K L
      j : RingHom K M
      k : RingHom K N
      f : RingHom L M
      g : RingHom L N
      p : Nat
      inst✝⁴ : ExpChar M p
      inst✝³ : ExpChar K p
      inst✝² : PerfectRing M p
      inst✝¹ : IsPRadical i p
      inst✝ : ExpChar L p
      x1 x2 : L
      n1 : Nat
      y1 : K
      h1 : Eq (i y1) (HPow.hPow x1 (HPow.hPow p n1))
      ⊢ Eq ((↑{ toFun := PerfectRing.liftAux i j p, map_one' := ⋯, map_mul' := ⋯ }). …
    -/
    obtain ⟨n2, y2, h2⟩ := IsPRadical.pow_mem i p x2
    /-
      case intro.intro.intro.intro
      K : Type u_1
      L : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁸ : CommRing K
      inst✝⁷ : CommRing L
      inst✝⁶ : CommRing M
      inst✝⁵ : CommRing N
      i : RingHom K L
      j : RingHom K M
      k : RingHom K N
      f : RingHom L M
      g : RingHom L N
      p : Nat
      inst✝⁴ : ExpChar M p
      inst✝³ : ExpChar K p
      inst✝² : PerfectRing M p
      inst✝¹ : IsPRadical i p
      inst✝ : ExpChar L p
      x1 x2 : L
      n1 : Nat
      y1 : K
      h1 : Eq (i y1) (HPow.hPow x1 (HPow.hPow p n1))
      n2 : Nat
      y2 : K
      h2 : Eq (i y2) (HPow.hPow x2 (HPow.hPow p n2))
      ⊢ Eq ((↑{ toFun := PerfectRing.liftAux i j p, map_one' := ⋯, map_mul' := ⋯ }). …
    -/
    simp only; rw [liftAux_apply i j p _ _ _ h1, liftAux_apply i j p _ _ _ h2,
      liftAux_apply i j p (x1 + x2) (n1 + n2) (y1 ^ p ^ n2 + y2 ^ p ^ n1) (by rw [map_add,
        map_pow, map_pow, h1, h2, ← pow_mul, ← pow_add, ← pow_mul, ← pow_add,
        add_comm n2, add_pow_expChar_pow]),
      map_add, map_pow, map_pow, map_add, ← iterateFrobeniusEquiv_def]
    /-
      case intro.intro.intro.intro
      K : Type u_1
      L : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁸ : CommRing K
      inst✝⁷ : CommRing L
      inst✝⁶ : CommRing M
      inst✝⁵ : CommRing N
      i : RingHom K L
      j : RingHom K M
      k : RingHom K N
      f : RingHom L M
      g : RingHom L N
      p : Nat
      inst✝⁴ : ExpChar M p
      inst✝³ : ExpChar K p
      inst✝² : PerfectRing M p
      inst✝¹ : IsPRadical i p
      inst✝ : ExpChar L p
      x1 x2 : L
      n1 : Nat
      y1 : K
      h1 : Eq (i y1) (HPow.hPow x1 (HPow.hPow p n1))
      n2 : Nat
      y2 : K
      h2 : Eq (i y2) (HPow.hPow x2 (HPow.hPow p n2))
      ⊢ Eq (HAdd.hAdd ((iterateFrobeniusEquiv M p (HAdd.hAdd n1 n2)).symm ((iterateF …
    -/
    nth_rw 1 [iterateFrobeniusEquiv_symm_add_apply]
    rw [RingEquiv.symm_apply_apply, add_comm n1, iterateFrobeniusEquiv_symm_add_apply,
      ← iterateFrobeniusEquiv_def, RingEquiv.symm_apply_apply]


theorem lift_apply (x : L) (n : ℕ) (y : K) (h : i y = x ^ p ^ n) :
    lift i j p x = (iterateFrobeniusEquiv M p n).symm (j y) :=
  liftAux_apply i j p _ _ _ h


@[simp]
theorem lift_comp_apply (x : K) : lift i j p (i x) = j x := by
  /-
    K : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁷ : CommRing K
    inst✝⁶ : CommRing L
    inst✝⁵ : CommRing M
    i : RingHom K L
    j : RingHom K M
    p : Nat
    inst✝⁴ : ExpChar M p
    inst✝³ : ExpChar K p
    inst✝² : PerfectRing M p
    inst✝¹ : IsPRadical i p
    inst✝ : ExpChar L p
    x : K
    ⊢ Eq ((PerfectRing.lift i j p) (i x)) (j x)
  -/
  rw [lift_apply i j p _ 0 x (by rw [pow_zero, pow_one]), iterateFrobeniusEquiv_zero]; rfl
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


@[simp]
theorem lift_comp : (lift i j p).comp i = j := RingHom.ext (lift_comp_apply i j p)


theorem lift_self_apply [PerfectRing L p] (x : L) : lift i i p x = x := liftAux_self_apply i p x


@[simp]
theorem lift_self [PerfectRing L p] : lift i i p = RingHom.id L :=
  RingHom.ext (liftAux_self_apply i p)


theorem lift_id_apply (x : K) : lift (RingHom.id K) j p x = j x := liftAux_id_apply j p x


@[simp]
theorem lift_id : lift (RingHom.id K) j p = j := RingHom.ext (liftAux_id_apply j p)


@[simp]
theorem comp_lift : lift i (f.comp i) p = f :=
  IsPRadical.injective_comp_of_perfect _ i p (lift_comp i _ p)


theorem comp_lift_apply (x : L) : lift i (f.comp i) p x = f x := congr($(comp_lift i f p) x)


variable (M) in
/-- If `i : K →+* L` is a homomorphisms of characteristic `p` rings, such that
`i` is `p`-radical, and `M` is a perfect ring of characteristic `p`,
then `K →+* M` is one-to-one correspondence to
`L →+* M`, given by `PerfectRing.lift`. This is a generalization to `PerfectClosure.lift`. -/
def liftEquiv : (K →+* M) ≃ (L →+* M) where
  toFun j := lift i j p
  invFun f := f.comp i
  left_inv f := lift_comp i f p
  right_inv f := comp_lift i f p


theorem liftEquiv_apply : liftEquiv M i p j = lift i j p := rfl


theorem liftEquiv_symm_apply : (liftEquiv M i p).symm f = f.comp i := rfl


theorem liftEquiv_id_apply : liftEquiv M (RingHom.id K) p j = j :=
  lift_id j p


@[simp]
theorem liftEquiv_id : liftEquiv M (RingHom.id K) p = Equiv.refl _ :=
  Equiv.ext (liftEquiv_id_apply · p)


@[simp]
theorem lift_comp_lift : (lift j k p).comp (lift i j p) = lift i k p :=
                                                 /-
                                                   K : Type u_1
                                                   L : Type u_2
                                                   M : Type u_3
                                                   N : Type u_4
                                                   inst✝¹¹ : CommRing K
                                                   inst✝¹⁰ : CommRing L
                                                   inst✝⁹ : CommRing M
                                                   inst✝⁸ : CommRing N
                                                   i : RingHom K L
                                                   j : RingHom K M
                                                   k : RingHom K N
                                                   p : Nat
                                                   inst✝⁷ : ExpChar M p
                                                   inst✝⁶ : ExpChar K p
                                                   inst✝⁵ : PerfectRing M p
                                                   inst✝⁴ : IsPRadical i p
                                                   inst✝³ : ExpChar L p
                                                   inst✝² : ExpChar N p
                                                   inst✝¹ : PerfectRing N p
                                                   inst✝ : IsPRadical j p
                                                   ⊢ Eq ((fun f => f.comp i) ((PerfectRing.lift j k p).comp (PerfectRing.lift i j …
                                                 -/
  IsPRadical.injective_comp_of_perfect _ i p (by ext; simp)
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
theorem lift_comp_lift_apply (x : L) : lift j k p (lift i j p x) = lift i k p x :=
  congr($(lift_comp_lift i j k p) x)


theorem lift_comp_lift_apply_eq_self [PerfectRing L p] (x : L) :
    lift j i p (lift i j p x) = x := by
  /-
    K : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁹ : CommRing K
    inst✝⁸ : CommRing L
    inst✝⁷ : CommRing M
    i : RingHom K L
    j : RingHom K M
    p : Nat
    inst✝⁶ : ExpChar M p
    inst✝⁵ : ExpChar K p
    inst✝⁴ : PerfectRing M p
    inst✝³ : IsPRadical i p
    inst✝² : ExpChar L p
    inst✝¹ : IsPRadical j p
    inst✝ : PerfectRing L p
    x : L
    ⊢ Eq ((PerfectRing.lift j i p) ((PerfectRing.lift i j p) x)) x
  -/
  rw [lift_comp_lift_apply, lift_self_apply]
  /-
    🎉 no goals
  -/


theorem lift_comp_lift_eq_id [PerfectRing L p] :
    (lift j i p).comp (lift i j p) = RingHom.id L :=
  RingHom.ext (lift_comp_lift_apply_eq_self i j p)


@[simp]
theorem lift_lift : lift g (lift i j p) p = lift (g.comp i) j p := by
  /-
    K : Type u_1
    L : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝¹¹ : CommRing K
    inst✝¹⁰ : CommRing L
    inst✝⁹ : CommRing M
    inst✝⁸ : CommRing N
    i : RingHom K L
    j : RingHom K M
    g : RingHom L N
    p : Nat
    inst✝⁷ : ExpChar M p
    inst✝⁶ : ExpChar K p
    inst✝⁵ : PerfectRing M p
    inst✝⁴ : IsPRadical i p
    inst✝³ : ExpChar L p
    inst✝² : ExpChar N p
    inst✝¹ : IsPRadical g p
    inst✝ : IsPRadical (g.comp i) p
    ⊢ Eq (PerfectRing.lift g (PerfectRing.lift i j p) p) (PerfectRing.lift (g.comp …
  -/
  refine IsPRadical.injective_comp_of_perfect _ (g.comp i) p ?_
  /-
    K : Type u_1
    L : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝¹¹ : CommRing K
    inst✝¹⁰ : CommRing L
    inst✝⁹ : CommRing M
    inst✝⁸ : CommRing N
    i : RingHom K L
    j : RingHom K M
    g : RingHom L N
    p : Nat
    inst✝⁷ : ExpChar M p
    inst✝⁶ : ExpChar K p
    inst✝⁵ : PerfectRing M p
    inst✝⁴ : IsPRadical i p
    inst✝³ : ExpChar L p
    inst✝² : ExpChar N p
    inst✝¹ : IsPRadical g p
    inst✝ : IsPRadical (g.comp i) p
    ⊢ Eq ((fun f => f.comp (g.comp i)) (PerfectRing.lift g (PerfectRing.lift i j p …
  -/
  simp_rw [← RingHom.comp_assoc _ _ (lift g _ p), lift_comp]
  /-
    🎉 no goals
  -/


theorem lift_lift_apply (x : N) : lift g (lift i j p) p x = lift (g.comp i) j p x :=
  congr($(lift_lift i j g p) x)


@[simp]
theorem liftEquiv_comp_apply :
    liftEquiv M g p (liftEquiv M i p j) = liftEquiv M (g.comp i) p j := lift_lift i j g p


@[simp]
theorem liftEquiv_trans :
    (liftEquiv M i p).trans (liftEquiv M g p) = liftEquiv M (g.comp i) p :=
  Equiv.ext (liftEquiv_comp_apply i · g p)


/-- If `L` and `M` are both perfect closures of `K`, then there is a ring isomorphism `L ≃+* M`.
This is similar to `IsAlgClosure.equiv` and `IsSepClosure.equiv`. -/
def equiv : L ≃+* M where
  __ := PerfectRing.lift i j p
  invFun := PerfectRing.liftAux j i p
  left_inv := PerfectRing.lift_comp_lift_apply_eq_self i j p
  right_inv := PerfectRing.lift_comp_lift_apply_eq_self j i p


theorem equiv_toRingHom : (equiv i j p).toRingHom = PerfectRing.lift i j p := rfl


@[simp]
theorem equiv_symm : (equiv i j p).symm = equiv j i p := rfl


theorem equiv_symm_toRingHom :
    (equiv i j p).symm.toRingHom = PerfectRing.lift j i p := rfl


theorem equiv_apply (x : L) (n : ℕ) (y : K) (h : i y = x ^ p ^ n) :
    equiv i j p x = (iterateFrobeniusEquiv M p n).symm (j y) :=
  PerfectRing.liftAux_apply i j p _ _ _ h


theorem equiv_symm_apply (x : M) (n : ℕ) (y : K) (h : j y = x ^ p ^ n) :
    (equiv i j p).symm x = (iterateFrobeniusEquiv L p n).symm (i y) := by
  /-
    K : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁹ : CommRing K
    inst✝⁸ : CommRing L
    inst✝⁷ : CommRing M
    i : RingHom K L
    j : RingHom K M
    p : Nat
    inst✝⁶ : ExpChar M p
    inst✝⁵ : ExpChar K p
    inst✝⁴ : ExpChar L p
    inst✝³ : PerfectRing L p
    inst✝² : IsPerfectClosure i p
    inst✝¹ : PerfectRing M p
    inst✝ : IsPerfectClosure j p
    x : M
    n : Nat
    y : K
    h : Eq (j y) (HPow.hPow x (HPow.hPow p n))
    ⊢ Eq ((IsPerfectClosure.equiv i j p).symm x) ((iterateFrobeniusEquiv L p n).sy …
  -/
  rw [equiv_symm, equiv_apply j i p _ _ _ h]
  /-
    🎉 no goals
  -/


theorem equiv_self_apply (x : L) : equiv i i p x = x :=
  PerfectRing.liftAux_self_apply i p x


@[simp]
theorem equiv_self : equiv i i p = RingEquiv.refl L :=
  RingEquiv.ext (equiv_self_apply i p)


@[simp]
theorem equiv_comp_apply (x : K) : equiv i j p (i x) = j x :=
  PerfectRing.lift_comp_apply i j p x


@[simp]
theorem equiv_comp : RingHom.comp (equiv i j p) i = j :=
  RingHom.ext (equiv_comp_apply i j p)


@[simp]
theorem equiv_comp_equiv_apply (x : L) :
    equiv j k p (equiv i j p x) = equiv i k p x :=
  PerfectRing.lift_comp_lift_apply i j k p x


@[simp]
theorem equiv_comp_equiv : (equiv i j p).trans (equiv j k p) = equiv i k p :=
  RingEquiv.ext (equiv_comp_equiv_apply i j k p)


theorem equiv_comp_equiv_apply_eq_self (x : L) :
    equiv j i p (equiv i j p x) = x := by
  /-
    K : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁹ : CommRing K
    inst✝⁸ : CommRing L
    inst✝⁷ : CommRing M
    i : RingHom K L
    j : RingHom K M
    p : Nat
    inst✝⁶ : ExpChar M p
    inst✝⁵ : ExpChar K p
    inst✝⁴ : ExpChar L p
    inst✝³ : PerfectRing L p
    inst✝² : IsPerfectClosure i p
    inst✝¹ : PerfectRing M p
    inst✝ : IsPerfectClosure j p
    x : L
    ⊢ Eq ((IsPerfectClosure.equiv j i p) ((IsPerfectClosure.equiv i j p) x)) x
  -/
  rw [equiv_comp_equiv_apply, equiv_self_apply]
  /-
    🎉 no goals
  -/


theorem equiv_comp_equiv_eq_id :
    (equiv i j p).trans (equiv j i p) = RingEquiv.refl L :=
  RingEquiv.ext (equiv_comp_equiv_apply_eq_self i j p)


/-- The absolute perfect closure `PerfectClosure` is a `p`-radical extension over the base ring.
In particular, it is a perfect closure of the base ring, that is,
`IsPerfectClosure (PerfectClosure.of K p) p`. -/
instance isPRadical : IsPRadical (PerfectClosure.of K p) p where
  pow_mem' x := PerfectClosure.induction_on x fun x ↦ ⟨x.1, x.2, by
    /-
      K : Type u_1
      L : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝² : CommRing K
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : CharP K p
      x✝ : PerfectClosure K p
      x : Prod Nat K
      ⊢ Eq ((PerfectClosure.of K p) x.2) (HPow.hPow (PerfectClosure.mk K p x) (HPow. …
    -/
    rw [← iterate_frobenius, iterate_frobenius_mk K p x.1 x.2]⟩
    /-
      🎉 no goals
    -/
  ker_le' x h := by
    /-
      K : Type u_1
      L : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝² : CommRing K
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : CharP K p
      x : K
      h : Membership.mem (RingHom.ker (PerfectClosure.of K p)) x
      ⊢ Membership.mem (pNilradical K p) x
    -/
    rw [RingHom.mem_ker, of_apply, zero_def, mk_eq_iff] at h
    /-
      K : Type u_1
      L : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝² : CommRing K
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : CharP K p
      x : K
      h : Exists fun z => Eq (Nat.iterate (⇑(frobenius K p)) (HAdd.hAdd { fst := 0,  …
      ⊢ Membership.mem (pNilradical K p) x
    -/
    obtain ⟨n, h⟩ := h
    /-
      case intro
      K : Type u_1
      L : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝² : CommRing K
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : CharP K p
      x : K
      n : Nat
      h : Eq (Nat.iterate (⇑(frobenius K p)) (HAdd.hAdd { fst := 0, snd := 0 }.1 n)  …
      ⊢ Membership.mem (pNilradical K p) x
    -/
    simp_rw [zero_add, ← coe_iterateFrobenius, map_zero] at h
    /-
      case intro
      K : Type u_1
      L : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝² : CommRing K
      p : Nat
      inst✝¹ : Fact (Nat.Prime p)
      inst✝ : CharP K p
      x : K
      n : Nat
      h : Eq ((iterateFrobenius K p n) x) 0
      ⊢ Membership.mem (pNilradical K p) x
    -/
    exact mem_pNilradical.2 ⟨n, h⟩
    /-
      🎉 no goals
    -/


/-- If `L / K` is a `p`-radical field extension, then it is purely inseparable. -/
theorem IsPRadical.isPurelyInseparable [IsPRadical (algebraMap K L) p] :
    IsPurelyInseparable K L :=
  (isPurelyInseparable_iff_pow_mem K p).2 (IsPRadical.pow_mem (algebraMap K L) p)


/-- If `L / K` is a purely inseparable field extension, then it is `p`-radical. In particular, if
`L` is perfect, then the (relative) perfect closure `perfectClosure K L` is a perfect closure
of `K`, that is, `IsPerfectClosure (algebraMap K (perfectClosure K L)) p`. -/
instance IsPurelyInseparable.isPRadical [IsPurelyInseparable K L] :
    IsPRadical (algebraMap K L) p where
  pow_mem' := (isPurelyInseparable_iff_pow_mem K p).1 ‹_›
  ker_le' := (RingHom.injective_iff_ker_eq_bot _).1 (algebraMap K L).injective ▸ bot_le


