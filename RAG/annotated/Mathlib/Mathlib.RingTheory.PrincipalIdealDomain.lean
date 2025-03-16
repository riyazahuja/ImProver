instance bot_isPrincipal : (⊥ : Submodule R M).IsPrincipal :=
          /-
            R : Type u
            M : Type v
            inst✝² : Semiring R
            inst✝¹ : AddCommGroup M
            inst✝ : Module R M
            ⊢ Eq Bot.bot (Submodule.span R (Singleton.singleton 0))
          -/
  ⟨⟨0, by simp⟩⟩
          /-
            🎉 no goals
          -/


instance top_isPrincipal : (⊤ : Submodule R R).IsPrincipal :=
  ⟨⟨1, Ideal.span_singleton_one.symm⟩⟩


/-- A Bézout ring is a ring whose finitely generated ideals are principal. -/
class IsBezout : Prop where
  /-- Any finitely generated ideal is principal. -/
  isPrincipal_of_FG : ∀ I : Ideal R, I.FG → I.IsPrincipal


instance (priority := 100) IsBezout.of_isPrincipalIdealRing [IsPrincipalIdealRing R] : IsBezout R :=
  ⟨fun I _ => IsPrincipalIdealRing.principal I⟩


instance (priority := 100) DivisionRing.isPrincipalIdealRing (K : Type u) [DivisionRing K] :
    IsPrincipalIdealRing K where
  principal S := by
    /-
      R : Type u
      M : Type v
      inst✝³ : Semiring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      K : Type u
      inst✝ : DivisionRing K
      S : Ideal K
      ⊢ Submodule.IsPrincipal S
    -/
    rcases Ideal.eq_bot_or_top S with (rfl | rfl)
      /-
        case inl
        R : Type u
        M : Type v
        inst✝³ : Semiring R
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        K : Type u
        inst✝ : DivisionRing K
        ⊢ Submodule.IsPrincipal Bot.bot
      -/
    · apply bot_isPrincipal
      /-
        🎉 no goals
      -/
      /-
        case inr
        R : Type u
        M : Type v
        inst✝³ : Semiring R
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        K : Type u
        inst✝ : DivisionRing K
        ⊢ Submodule.IsPrincipal Top.top
      -/
    · apply top_isPrincipal
      /-
        🎉 no goals
      -/


/-- `generator I`, if `I` is a principal submodule, is an `x ∈ M` such that `span R {x} = I` -/
noncomputable def generator (S : Submodule R M) [S.IsPrincipal] : M :=
  Classical.choose (principal S)


theorem span_singleton_generator (S : Submodule R M) [S.IsPrincipal] : span R {generator S} = S :=
  Eq.symm (Classical.choose_spec (principal S))


@[simp]
theorem _root_.Ideal.span_singleton_generator (I : Ideal R) [I.IsPrincipal] :
    Ideal.span ({generator I} : Set R) = I :=
  Eq.symm (Classical.choose_spec (principal I))


@[simp]
theorem generator_mem (S : Submodule R M) [S.IsPrincipal] : generator S ∈ S := by
  /-
    R : Type u
    M : Type v
    inst✝³ : AddCommMonoid M
    inst✝² : Semiring R
    inst✝¹ : Module R M
    S : Submodule R M
    inst✝ : S.IsPrincipal
    ⊢ Membership.mem S (Submodule.IsPrincipal.generator S)
  -/
  have : generator S ∈ span R {generator S} := subset_span (mem_singleton _)
  /-
    R : Type u
    M : Type v
    inst✝³ : AddCommMonoid M
    inst✝² : Semiring R
    inst✝¹ : Module R M
    S : Submodule R M
    inst✝ : S.IsPrincipal
    this : Membership.mem (Submodule.span R (Singleton.singleton (Submodule.IsPrin …
    ⊢ Membership.mem S (Submodule.IsPrincipal.generator S)
  -/
  convert this
  /-
    case h.e'_4
    R : Type u
    M : Type v
    inst✝³ : AddCommMonoid M
    inst✝² : Semiring R
    inst✝¹ : Module R M
    S : Submodule R M
    inst✝ : S.IsPrincipal
    this : Membership.mem (Submodule.span R (Singleton.singleton (Submodule.IsPrin …
    ⊢ Eq S (Submodule.span R (Singleton.singleton (Submodule.IsPrincipal.generator …
  -/
  exact span_singleton_generator S |>.symm
  /-
    🎉 no goals
  -/


theorem mem_iff_eq_smul_generator (S : Submodule R M) [S.IsPrincipal] {x : M} :
    x ∈ S ↔ ∃ s : R, x = s • generator S := by
  /-
    R : Type u
    M : Type v
    inst✝³ : AddCommMonoid M
    inst✝² : Semiring R
    inst✝¹ : Module R M
    S : Submodule R M
    inst✝ : S.IsPrincipal
    x : M
    ⊢ Iff (Membership.mem S x) (Exists fun s => Eq x (HSMul.hSMul s (Submodule.IsP …
  -/
  simp_rw [@eq_comm _ x, ← mem_span_singleton, span_singleton_generator]
  /-
    🎉 no goals
  -/


theorem eq_bot_iff_generator_eq_zero (S : Submodule R M) [S.IsPrincipal] :
                                  /-
                                    R : Type u
                                    M : Type v
                                    inst✝³ : AddCommMonoid M
                                    inst✝² : Semiring R
                                    inst✝¹ : Module R M
                                    S : Submodule R M
                                    inst✝ : S.IsPrincipal
                                    ⊢ Iff (Eq S Bot.bot) (Eq (Submodule.IsPrincipal.generator S) 0)
                                  -/
    S = ⊥ ↔ generator S = 0 := by rw [← @span_singleton_eq_bot R M, span_singleton_generator]
                                  /-
                                    🎉 no goals
                                  -/


protected lemma fg {S : Submodule R M} (h : S.IsPrincipal) : S.FG :=
                     /-
                       R : Type u
                       M : Type v
                       inst✝² : AddCommMonoid M
                       inst✝¹ : Semiring R
                       inst✝ : Module R M
                       S : Submodule R M
                       h : S.IsPrincipal
                       ⊢ Eq (Submodule.span R ↑(Singleton.singleton (Submodule.IsPrincipal.generator  …
                     -/
  ⟨{h.generator}, by simp only [Finset.coe_singleton, span_singleton_generator]⟩
                     /-
                       🎉 no goals
                     -/

-- See note [lower instance priority]

instance (priority := 100) _root_.PrincipalIdealRing.isNoetherianRing [IsPrincipalIdealRing R] :
    IsNoetherianRing R where
  noetherian S := (IsPrincipalIdealRing.principal S).fg

-- See note [lower instance priority]

instance (priority := 100) _root_.IsPrincipalIdealRing.of_isNoetherianRing_of_isBezout
    [IsNoetherianRing R] [IsBezout R] : IsPrincipalIdealRing R where
  principal S := IsBezout.isPrincipal_of_FG S (IsNoetherian.noetherian S)


theorem associated_generator_span_self [IsPrincipalIdealRing R] [IsDomain R] (r : R) :
    Associated (generator <| Ideal.span {r}) r := by
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsPrincipalIdealRing R
    inst✝ : IsDomain R
    r : R
    ⊢ Associated (Submodule.IsPrincipal.generator (Ideal.span (Singleton.singleton …
  -/
  rw [← Ideal.span_singleton_eq_span_singleton]
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsPrincipalIdealRing R
    inst✝ : IsDomain R
    r : R
    ⊢ Eq (Ideal.span (Singleton.singleton (Submodule.IsPrincipal.generator (Ideal. …
  -/
  exact Ideal.span_singleton_generator _
  /-
    🎉 no goals
  -/


theorem mem_iff_generator_dvd (S : Ideal R) [S.IsPrincipal] {x : R} : x ∈ S ↔ generator S ∣ x :=
                                                                /-
                                                                  R : Type u
                                                                  inst✝¹ : CommRing R
                                                                  S : Ideal R
                                                                  inst✝ : Submodule.IsPrincipal S
                                                                  x a : R
                                                                  ⊢ Iff (Eq x (HSMul.hSMul a (Submodule.IsPrincipal.generator S))) (Eq x (HMul.h …
                                                                -/
  (mem_iff_eq_smul_generator S).trans (exists_congr fun a => by simp only [mul_comm, smul_eq_mul])
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem prime_generator_of_isPrime (S : Ideal R) [S.IsPrincipal] [is_prime : S.IsPrime]
    (ne_bot : S ≠ ⊥) : Prime (generator S) :=
  ⟨fun h => ne_bot ((eq_bot_iff_generator_eq_zero S).2 h), fun h =>
    is_prime.ne_top (S.eq_top_of_isUnit_mem (generator_mem S) h), fun _ _ => by
    /-
      R : Type u
      inst✝¹ : CommRing R
      S : Ideal R
      inst✝ : Submodule.IsPrincipal S
      is_prime : S.IsPrime
      ne_bot : Ne S Bot.bot
      x✝¹ x✝ : R
      ⊢ Dvd.dvd (Submodule.IsPrincipal.generator S) (HMul.hMul x✝¹ x✝) → Or (Dvd.dvd …
    -/
    simpa only [← mem_iff_generator_dvd S] using is_prime.2⟩
    /-
      🎉 no goals
    -/

-- Note that the converse may not hold if `ϕ` is not injective.

theorem generator_map_dvd_of_mem {N : Submodule R M} (ϕ : M →ₗ[R] R) [(N.map ϕ).IsPrincipal] {x : M}
    (hx : x ∈ N) : generator (N.map ϕ) ∣ ϕ x := by
  /-
    R : Type u
    M : Type v
    inst✝³ : AddCommMonoid M
    inst✝² : CommRing R
    inst✝¹ : Module R M
    N : Submodule R M
    ϕ : LinearMap (RingHom.id R) M R
    inst✝ : (Submodule.map ϕ N).IsPrincipal
    x : M
    hx : Membership.mem N x
    ⊢ Dvd.dvd (Submodule.IsPrincipal.generator (Submodule.map ϕ N)) (ϕ x)
  -/
  rw [← mem_iff_generator_dvd, Submodule.mem_map]
  /-
    R : Type u
    M : Type v
    inst✝³ : AddCommMonoid M
    inst✝² : CommRing R
    inst✝¹ : Module R M
    N : Submodule R M
    ϕ : LinearMap (RingHom.id R) M R
    inst✝ : (Submodule.map ϕ N).IsPrincipal
    x : M
    hx : Membership.mem N x
    ⊢ Exists fun y => And (Membership.mem N y) (Eq (ϕ y) (ϕ x))
  -/
  exact ⟨x, hx, rfl⟩
  /-
    🎉 no goals
  -/

-- Note that the converse may not hold if `ϕ` is not injective.

theorem generator_submoduleImage_dvd_of_mem {N O : Submodule R M} (hNO : N ≤ O) (ϕ : O →ₗ[R] R)
    [(ϕ.submoduleImage N).IsPrincipal] {x : M} (hx : x ∈ N) :
    generator (ϕ.submoduleImage N) ∣ ϕ ⟨x, hNO hx⟩ := by
  /-
    R : Type u
    M : Type v
    inst✝³ : AddCommMonoid M
    inst✝² : CommRing R
    inst✝¹ : Module R M
    N O : Submodule R M
    hNO : LE.le N O
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R
    inst✝ : (ϕ.submoduleImage N).IsPrincipal
    x : M
    hx : Membership.mem N x
    ⊢ Dvd.dvd (Submodule.IsPrincipal.generator (ϕ.submoduleImage N)) (ϕ ⟨x, ⋯⟩)
  -/
  rw [← mem_iff_generator_dvd, LinearMap.mem_submoduleImage_of_le hNO]
  /-
    R : Type u
    M : Type v
    inst✝³ : AddCommMonoid M
    inst✝² : CommRing R
    inst✝¹ : Module R M
    N O : Submodule R M
    hNO : LE.le N O
    ϕ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem O x) R
    inst✝ : (ϕ.submoduleImage N).IsPrincipal
    x : M
    hx : Membership.mem N x
    ⊢ Exists fun y => Exists fun yN => Eq (ϕ ⟨y, ⋯⟩) (ϕ ⟨x, ⋯⟩)
  -/
  exact ⟨x, hx, rfl⟩
  /-
    🎉 no goals
  -/


instance span_pair_isPrincipal [IsBezout R] (x y : R) : (Ideal.span {x, y}).IsPrincipal := by
  /-
    R : Type u
    M : Type v
    inst✝¹ : Ring R
    inst✝ : IsBezout R
    x y : R
    ⊢ Submodule.IsPrincipal (Ideal.span (Insert.insert x (Singleton.singleton y)))
  -/
  classical exact isPrincipal_of_FG (Ideal.span {x, y}) ⟨{x, y}, by simp⟩
  /-
    🎉 no goals
  -/


/-- A choice of gcd of two elements in a Bézout domain.

Note that the choice is usually not unique. -/
noncomputable def gcd : R := Submodule.IsPrincipal.generator (Ideal.span {x, y})


theorem span_gcd : Ideal.span {gcd x y} = Ideal.span {x, y} :=
  Ideal.span_singleton_generator _


theorem gcd_dvd_left : gcd x y ∣ x :=
                                                                            /-
                                                                              R : Type u
                                                                              inst✝¹ : CommRing R
                                                                              x y : R
                                                                              inst✝ : Submodule.IsPrincipal (Ideal.span (Insert.insert x (Singleton.singleto …
                                                                              ⊢ Membership.mem (Insert.insert x (Singleton.singleton y)) x
                                                                            -/
  (Submodule.IsPrincipal.mem_iff_generator_dvd _).mp (Ideal.subset_span (by simp))
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


theorem gcd_dvd_right : gcd x y ∣ y :=
                                                                            /-
                                                                              R : Type u
                                                                              inst✝¹ : CommRing R
                                                                              x y : R
                                                                              inst✝ : Submodule.IsPrincipal (Ideal.span (Insert.insert x (Singleton.singleto …
                                                                              ⊢ Membership.mem (Insert.insert x (Singleton.singleton y)) y
                                                                            -/
  (Submodule.IsPrincipal.mem_iff_generator_dvd _).mp (Ideal.subset_span (by simp))
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


variable {x y z} in
theorem dvd_gcd (hx : z ∣ x) (hy : z ∣ y) : z ∣ gcd x y := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    x y z : R
    inst✝ : Submodule.IsPrincipal (Ideal.span (Insert.insert x (Singleton.singleto …
    hx : Dvd.dvd z x
    hy : Dvd.dvd z y
    ⊢ Dvd.dvd z (IsBezout.gcd x y)
  -/
  rw [← Ideal.span_singleton_le_span_singleton] at hx hy ⊢
  /-
    R : Type u
    inst✝¹ : CommRing R
    x y z : R
    inst✝ : Submodule.IsPrincipal (Ideal.span (Insert.insert x (Singleton.singleto …
    hx : LE.le (Ideal.span (Singleton.singleton x)) (Ideal.span (Singleton.singlet …
    hy : LE.le (Ideal.span (Singleton.singleton y)) (Ideal.span (Singleton.singlet …
    ⊢ LE.le (Ideal.span (Singleton.singleton (IsBezout.gcd x y))) (Ideal.span (Sin …
  -/
  rw [span_gcd, Ideal.span_insert, sup_le_iff]
  /-
    R : Type u
    inst✝¹ : CommRing R
    x y z : R
    inst✝ : Submodule.IsPrincipal (Ideal.span (Insert.insert x (Singleton.singleto …
    hx : LE.le (Ideal.span (Singleton.singleton x)) (Ideal.span (Singleton.singlet …
    hy : LE.le (Ideal.span (Singleton.singleton y)) (Ideal.span (Singleton.singlet …
    ⊢ And (LE.le (Ideal.span (Singleton.singleton x)) (Ideal.span (Singleton.singl …
  -/
  exact ⟨hx, hy⟩
  /-
    🎉 no goals
  -/


theorem gcd_eq_sum : ∃ a b : R, a * x + b * y = gcd x y :=
                             /-
                               R : Type u
                               inst✝¹ : CommRing R
                               x y : R
                               inst✝ : Submodule.IsPrincipal (Ideal.span (Insert.insert x (Singleton.singleto …
                               ⊢ Membership.mem (Ideal.span (Insert.insert x (Singleton.singleton y))) (IsBez …
                             -/
  Ideal.mem_span_pair.mp (by rw [← span_gcd]; apply Ideal.subset_span; simp)
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem _root_.IsRelPrime.isCoprime (h : IsRelPrime x y) : IsCoprime x y := by
  rw [← Ideal.isCoprime_span_singleton_iff, Ideal.isCoprime_iff_sup_eq, ← Ideal.span_union,
    Set.singleton_union, ← span_gcd, Ideal.span_singleton_eq_top]
  /-
    R : Type u
    inst✝¹ : CommRing R
    x y : R
    inst✝ : Submodule.IsPrincipal (Ideal.span (Insert.insert x (Singleton.singleto …
    h : IsRelPrime x y
    ⊢ IsUnit (IsBezout.gcd x y)
  -/
  exact h (gcd_dvd_left x y) (gcd_dvd_right x y)
  /-
    🎉 no goals
  -/


theorem _root_.isRelPrime_iff_isCoprime : IsRelPrime x y ↔ IsCoprime x y :=
  ⟨IsRelPrime.isCoprime, IsCoprime.isRelPrime⟩


/-- Any Bézout domain is a GCD domain. This is not an instance since `GCDMonoid` contains data,
and this might not be how we would like to construct it. -/
noncomputable def toGCDDomain [IsBezout R] [IsDomain R] [DecidableEq R] : GCDMonoid R :=
  gcdMonoidOfGCD (gcd · ·) (gcd_dvd_left · ·) (gcd_dvd_right · ·) dvd_gcd


instance nonemptyGCDMonoid [IsBezout R] [IsDomain R] : Nonempty (GCDMonoid R) := by
  /-
    R : Type u
    M : Type v
    inst✝³ : CommRing R
    x y z : R
    inst✝² : Submodule.IsPrincipal (Ideal.span (Insert.insert x (Singleton.singlet …
    inst✝¹ : IsBezout R
    inst✝ : IsDomain R
    ⊢ Nonempty (GCDMonoid R)
  -/
  classical exact ⟨toGCDDomain R⟩
  /-
    🎉 no goals
  -/


theorem associated_gcd_gcd [IsDomain R] [GCDMonoid R] :
    Associated (IsBezout.gcd x y) (GCDMonoid.gcd x y) :=
  gcd_greatest_associated (gcd_dvd_left _ _ ) (gcd_dvd_right _ _) (fun _ => dvd_gcd)


theorem to_maximal_ideal [CommRing R] [IsDomain R] [IsPrincipalIdealRing R] {S : Ideal R}
    [hpi : IsPrime S] (hS : S ≠ ⊥) : IsMaximal S :=
  isMaximal_iff.2
    ⟨(ne_top_iff_one S).1 hpi.1, by
      /-
        R : Type u
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        inst✝ : IsPrincipalIdealRing R
        S : Ideal R
        hpi : S.IsPrime
        hS : Ne S Bot.bot
        ⊢ ∀ (J : Ideal R) (x : R), LE.le S J → Not (Membership.mem S x) → Membership.m …
      -/
      intro T x hST hxS hxT
      /-
        R : Type u
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        inst✝ : IsPrincipalIdealRing R
        S : Ideal R
        hpi : S.IsPrime
        hS : Ne S Bot.bot
        T : Ideal R
        x : R
        hST : LE.le S T
        hxS : Not (Membership.mem S x)
        hxT : Membership.mem T x
        ⊢ Membership.mem T 1
      -/
      cases' (mem_iff_generator_dvd _).1 (hST <| generator_mem S) with z hz
      cases hpi.mem_or_mem (show generator T * z ∈ S from hz ▸ generator_mem S) with
      | inl h =>
        have hTS : T ≤ S := by
          rwa [← T.span_singleton_generator, Ideal.span_le, singleton_subset_iff]
        exact (hxS <| hTS hxT).elim
      | inr h =>
        cases' (mem_iff_generator_dvd _).1 h with y hy
        have : generator S ≠ 0 := mt (eq_bot_iff_generator_eq_zero _).2 hS
        rw [← mul_one (generator S), hy, mul_left_comm, mul_right_inj' this] at hz
        exact hz.symm ▸ T.mul_mem_right _ (generator_mem T)⟩


theorem mod_mem_iff {S : Ideal R} {x y : R} (hy : y ∈ S) : x % y ∈ S ↔ x ∈ S :=
  ⟨fun hxy => div_add_mod x y ▸ S.add_mem (S.mul_mem_right _ hy) hxy, fun hx =>
    (mod_eq_sub_mul_div x y).symm ▸ S.sub_mem hx (S.mul_mem_right _ hy)⟩

-- see Note [lower instance priority]

instance (priority := 100) EuclideanDomain.to_principal_ideal_domain : IsPrincipalIdealRing R where
  principal S := by classical exact
    ⟨if h : { x : R | x ∈ S ∧ x ≠ 0 }.Nonempty then
        have wf : WellFounded (EuclideanDomain.r : R → R → Prop) := EuclideanDomain.r_wellFounded
        have hmin : WellFounded.min wf { x : R | x ∈ S ∧ x ≠ 0 } h ∈ S ∧
            WellFounded.min wf { x : R | x ∈ S ∧ x ≠ 0 } h ≠ 0 :=
          WellFounded.min_mem wf { x : R | x ∈ S ∧ x ≠ 0 } h
        ⟨WellFounded.min wf { x : R | x ∈ S ∧ x ≠ 0 } h,
          Submodule.ext fun x => ⟨fun hx =>
            div_add_mod x (WellFounded.min wf { x : R | x ∈ S ∧ x ≠ 0 } h) ▸
              (Ideal.mem_span_singleton.2 <| dvd_add (dvd_mul_right _ _) <| by
                have : x % WellFounded.min wf { x : R | x ∈ S ∧ x ≠ 0 } h ∉
                    { x : R | x ∈ S ∧ x ≠ 0 } :=
                  fun h₁ => WellFounded.not_lt_min wf _ h h₁ (mod_lt x hmin.2)
                have : x % WellFounded.min wf { x : R | x ∈ S ∧ x ≠ 0 } h = 0 := by
                  simp only [not_and_or, Set.mem_setOf_eq, not_ne_iff] at this
                  exact this.neg_resolve_left <| (mod_mem_iff hmin.1).2 hx
                simp [*]),
              fun hx =>
                let ⟨y, hy⟩ := Ideal.mem_span_singleton.1 hx
                hy.symm ▸ S.mul_mem_right _ hmin.1⟩⟩
      else ⟨0, Submodule.ext fun a => by
            rw [← @Submodule.bot_coe R R _ _ _, span_eq, Submodule.mem_bot]
            exact ⟨fun haS => by_contra fun ha0 => h ⟨a, ⟨haS, ha0⟩⟩,
              fun h₁ => h₁.symm ▸ S.zero_mem⟩⟩⟩


theorem IsField.isPrincipalIdealRing {R : Type*} [CommRing R] (h : IsField R) :
    IsPrincipalIdealRing R :=
  @EuclideanDomain.to_principal_ideal_domain R (@Field.toEuclideanDomain R h.toField)


theorem isMaximal_of_irreducible [CommRing R] [IsPrincipalIdealRing R] {p : R}
    (hp : Irreducible p) : Ideal.IsMaximal (span R ({p} : Set R)) :=
  ⟨⟨mt Ideal.span_singleton_eq_top.1 hp.1, fun I hI => by
      /-
        R : Type u
        inst✝¹ : CommRing R
        inst✝ : IsPrincipalIdealRing R
        p : R
        hp : Irreducible p
        I : Ideal R
        hI : LT.lt (Submodule.span R (Singleton.singleton p)) I
        ⊢ Eq I Top.top
      -/
      rcases principal I with ⟨a, rfl⟩
      /-
        case mk.intro
        R : Type u
        inst✝¹ : CommRing R
        inst✝ : IsPrincipalIdealRing R
        p : R
        hp : Irreducible p
        a : R
        hI : LT.lt (Submodule.span R (Singleton.singleton p)) (Submodule.span R (Singl …
        ⊢ Eq (Submodule.span R (Singleton.singleton a)) Top.top
      -/
      erw [Ideal.span_singleton_eq_top]
      /-
        case mk.intro
        R : Type u
        inst✝¹ : CommRing R
        inst✝ : IsPrincipalIdealRing R
        p : R
        hp : Irreducible p
        a : R
        hI : LT.lt (Submodule.span R (Singleton.singleton p)) (Submodule.span R (Singl …
        ⊢ IsUnit a
      -/
      rcases Ideal.span_singleton_le_span_singleton.1 (le_of_lt hI) with ⟨b, rfl⟩
      /-
        case mk.intro.intro
        R : Type u
        inst✝¹ : CommRing R
        inst✝ : IsPrincipalIdealRing R
        a b : R
        hp : Irreducible (HMul.hMul a b)
        hI : LT.lt (Submodule.span R (Singleton.singleton (HMul.hMul a b))) (Submodule …
        ⊢ IsUnit a
      -/
      refine (of_irreducible_mul hp).resolve_right (mt (fun hb => ?_) (not_le_of_lt hI))
      /-
        case mk.intro.intro
        R : Type u
        inst✝¹ : CommRing R
        inst✝ : IsPrincipalIdealRing R
        a b : R
        hp : Irreducible (HMul.hMul a b)
        hI : LT.lt (Submodule.span R (Singleton.singleton (HMul.hMul a b))) (Submodule …
        hb : IsUnit b
        ⊢ LE.le (Submodule.span R (Singleton.singleton a)) (Submodule.span R (Singleto …
      -/
      erw [Ideal.span_singleton_le_span_singleton, IsUnit.mul_right_dvd hb]⟩⟩
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-02-12")]
protected alias irreducible_iff_prime := irreducible_iff_prime


@[deprecated (since := "2024-02-12")]
protected alias associates_irreducible_iff_prime := associates_irreducible_iff_prime


/-- `factors a` is a multiset of irreducible elements whose product is `a`, up to units -/
noncomputable def factors (a : R) : Multiset R :=
  if h : a = 0 then ∅ else Classical.choose (WfDvdMonoid.exists_factors a h)


theorem factors_spec (a : R) (h : a ≠ 0) :
    (∀ b ∈ factors a, Irreducible b) ∧ Associated (factors a).prod a := by
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsPrincipalIdealRing R
    a : R
    h : Ne a 0
    ⊢ And (∀ (b : R), Membership.mem (PrincipalIdealRing.factors a) b → Irreducibl …
  -/
  unfold factors; rw [dif_neg h]
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsPrincipalIdealRing R
    a : R
    h : Ne a 0
    ⊢ And (∀ (b : R), Membership.mem (Classical.choose ⋯) b → Irreducible b) (Asso …
  -/
  exact Classical.choose_spec (WfDvdMonoid.exists_factors a h)
  /-
    🎉 no goals
  -/


theorem ne_zero_of_mem_factors {R : Type v} [CommRing R] [IsDomain R] [IsPrincipalIdealRing R]
    {a b : R} (ha : a ≠ 0) (hb : b ∈ factors a) : b ≠ 0 :=
  Irreducible.ne_zero ((factors_spec a ha).1 b hb)


theorem mem_submonoid_of_factors_subset_of_units_subset (s : Submonoid R) {a : R} (ha : a ≠ 0)
    (hfac : ∀ b ∈ factors a, b ∈ s) (hunit : ∀ c : Rˣ, (c : R) ∈ s) : a ∈ s := by
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsPrincipalIdealRing R
    s : Submonoid R
    a : R
    ha : Ne a 0
    hfac : ∀ (b : R), Membership.mem (PrincipalIdealRing.factors a) b → Membership …
    hunit : ∀ (c : Units R), Membership.mem s ↑c
    ⊢ Membership.mem s a
  -/
  rcases (factors_spec a ha).2 with ⟨c, hc⟩
  /-
    case intro
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsPrincipalIdealRing R
    s : Submonoid R
    a : R
    ha : Ne a 0
    hfac : ∀ (b : R), Membership.mem (PrincipalIdealRing.factors a) b → Membership …
    hunit : ∀ (c : Units R), Membership.mem s ↑c
    c : Units R
    hc : Eq (HMul.hMul (PrincipalIdealRing.factors a).prod ↑c) a
    ⊢ Membership.mem s a
  -/
  rw [← hc]
  /-
    case intro
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : IsPrincipalIdealRing R
    s : Submonoid R
    a : R
    ha : Ne a 0
    hfac : ∀ (b : R), Membership.mem (PrincipalIdealRing.factors a) b → Membership …
    hunit : ∀ (c : Units R), Membership.mem s ↑c
    c : Units R
    hc : Eq (HMul.hMul (PrincipalIdealRing.factors a).prod ↑c) a
    ⊢ Membership.mem s (HMul.hMul (PrincipalIdealRing.factors a).prod ↑c)
  -/
  exact mul_mem (multiset_prod_mem _ hfac) (hunit _)
  /-
    🎉 no goals
  -/


/-- If a `RingHom` maps all units and all factors of an element `a` into a submonoid `s`, then it
also maps `a` into that submonoid. -/
theorem ringHom_mem_submonoid_of_factors_subset_of_units_subset {R S : Type*} [CommRing R]
    [IsDomain R] [IsPrincipalIdealRing R] [Semiring S] (f : R →+* S) (s : Submonoid S) (a : R)
    (ha : a ≠ 0) (h : ∀ b ∈ factors a, f b ∈ s) (hf : ∀ c : Rˣ, f c ∈ s) : f a ∈ s :=
  mem_submonoid_of_factors_subset_of_units_subset (s.comap f.toMonoidHom) ha h hf

-- see Note [lower instance priority]

/-- A principal ideal domain has unique factorization -/
instance (priority := 100) to_uniqueFactorizationMonoid : UniqueFactorizationMonoid R :=
  { (IsNoetherianRing.wfDvdMonoid : WfDvdMonoid R) with
    irreducible_iff_prime := irreducible_iff_prime }


theorem Submodule.IsPrincipal.map (f : M →ₗ[R] N) {S : Submodule R M}
    (hI : IsPrincipal S) : IsPrincipal (map f S) :=
  ⟨⟨f (IsPrincipal.generator S), by
      /-
        R : Type u
        M : Type v
        N : Type u_2
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : AddCommGroup N
        inst✝¹ : Module R M
        inst✝ : Module R N
        f : LinearMap (RingHom.id R) M N
        S : Submodule R M
        hI : S.IsPrincipal
        ⊢ Eq (Submodule.map f S) (Submodule.span R (Singleton.singleton (f (Submodule. …
      -/
      rw [← Set.image_singleton, ← map_span, span_singleton_generator]⟩⟩
      /-
        🎉 no goals
      -/


theorem Submodule.IsPrincipal.of_comap (f : M →ₗ[R] N) (hf : Function.Surjective f)
    (S : Submodule R N) [hI : IsPrincipal (S.comap f)] : IsPrincipal S := by
  /-
    R : Type u
    M : Type v
    N : Type u_2
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    hf : Function.Surjective ⇑f
    S : Submodule R N
    hI : (Submodule.comap f S).IsPrincipal
    ⊢ S.IsPrincipal
  -/
  rw [← Submodule.map_comap_eq_of_surjective hf S]
  /-
    R : Type u
    M : Type v
    N : Type u_2
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    hf : Function.Surjective ⇑f
    S : Submodule R N
    hI : (Submodule.comap f S).IsPrincipal
    ⊢ (Submodule.map f (Submodule.comap f S)).IsPrincipal
  -/
  exact hI.map f
  /-
    🎉 no goals
  -/


theorem Submodule.IsPrincipal.map_ringHom (f : F) {I : Ideal R}
    (hI : IsPrincipal I) : IsPrincipal (Ideal.map f I) :=
  ⟨⟨f (IsPrincipal.generator I), by
      rw [Ideal.submodule_span_eq, ← Set.image_singleton, ← Ideal.map_span,
      Ideal.span_singleton_generator]⟩⟩


theorem Ideal.IsPrincipal.of_comap (f : F) (hf : Function.Surjective f) (I : Ideal S)
    [hI : IsPrincipal (I.comap f)] : IsPrincipal I := by
  /-
    R : Type u
    S : Type u_1
    F : Type u_3
    inst✝³ : Ring R
    inst✝² : Ring S
    inst✝¹ : FunLike F R S
    inst✝ : RingHomClass F R S
    f : F
    hf : Function.Surjective ⇑f
    I : Ideal S
    hI : Submodule.IsPrincipal (Ideal.comap f I)
    ⊢ Submodule.IsPrincipal I
  -/
  rw [← map_comap_of_surjective f hf I]
  /-
    R : Type u
    S : Type u_1
    F : Type u_3
    inst✝³ : Ring R
    inst✝² : Ring S
    inst✝¹ : FunLike F R S
    inst✝ : RingHomClass F R S
    f : F
    hf : Function.Surjective ⇑f
    I : Ideal S
    hI : Submodule.IsPrincipal (Ideal.comap f I)
    ⊢ Submodule.IsPrincipal (Ideal.map f (Ideal.comap f I))
  -/
  exact hI.map_ringHom f
  /-
    🎉 no goals
  -/


/-- The surjective image of a principal ideal ring is again a principal ideal ring. -/
theorem IsPrincipalIdealRing.of_surjective [IsPrincipalIdealRing R] (f : F)
    (hf : Function.Surjective f) : IsPrincipalIdealRing S :=
  ⟨fun I => Ideal.IsPrincipal.of_comap f hf I⟩


theorem isCoprime_of_dvd (x y : R) (nonzero : ¬(x = 0 ∧ y = 0))
    (H : ∀ z ∈ nonunits R, z ≠ 0 → z ∣ x → ¬z ∣ y) : IsCoprime x y :=
  (isRelPrime_of_no_nonunits_factors nonzero H).isCoprime


theorem dvd_or_coprime (x y : R) (h : Irreducible x) : x ∣ y ∨ IsCoprime x y :=
  h.dvd_or_isRelPrime.imp_right IsRelPrime.isCoprime


/-- See also `Irreducible.isRelPrime_iff_not_dvd`. -/
theorem Irreducible.coprime_iff_not_dvd {p n : R} (hp : Irreducible p) :
                                 /-
                                   R : Type u
                                   inst✝¹ : CommRing R
                                   inst✝ : IsBezout R
                                   p n : R
                                   hp : Irreducible p
                                   ⊢ Iff (IsCoprime p n) (Not (Dvd.dvd p n))
                                 -/
    IsCoprime p n ↔ ¬p ∣ n := by rw [← isRelPrime_iff_isCoprime, hp.isRelPrime_iff_not_dvd]
                                 /-
                                   🎉 no goals
                                 -/


/-- See also `Irreducible.coprime_iff_not_dvd'`. -/
theorem Irreducible.dvd_iff_not_coprime {p n : R} (hp : Irreducible p) : p ∣ n ↔ ¬IsCoprime p n :=
  iff_not_comm.2 hp.coprime_iff_not_dvd


theorem Irreducible.coprime_pow_of_not_dvd {p a : R} (m : ℕ) (hp : Irreducible p) (h : ¬p ∣ a) :
    IsCoprime a (p ^ m) :=
  (hp.coprime_iff_not_dvd.2 h).symm.pow_right


theorem Irreducible.coprime_or_dvd {p : R} (hp : Irreducible p) (i : R) : IsCoprime p i ∨ p ∣ i :=
  (_root_.em _).imp_right hp.dvd_iff_not_coprime.2


theorem IsBezout.span_gcd_eq_span_gcd (x y : R) :
    span {GCDMonoid.gcd x y} = span {IsBezout.gcd x y} := by
  /-
    R : Type u
    inst✝³ : CommRing R
    inst✝² : IsBezout R
    inst✝¹ : IsDomain R
    inst✝ : GCDMonoid R
    x y : R
    ⊢ Eq (Ideal.span (Singleton.singleton (GCDMonoid.gcd x y))) (Ideal.span (Singl …
  -/
  rw [Ideal.span_singleton_eq_span_singleton]
  exact associated_of_dvd_dvd
    (IsBezout.dvd_gcd (GCDMonoid.gcd_dvd_left _ _) <| GCDMonoid.gcd_dvd_right _ _)
    (GCDMonoid.dvd_gcd (IsBezout.gcd_dvd_left _ _) <| IsBezout.gcd_dvd_right _ _)


theorem span_gcd (x y : R) : span {gcd x y} = span {x, y} := by
  /-
    R : Type u
    inst✝³ : CommRing R
    inst✝² : IsBezout R
    inst✝¹ : IsDomain R
    inst✝ : GCDMonoid R
    x y : R
    ⊢ Eq (Ideal.span (Singleton.singleton (GCDMonoid.gcd x y))) (Ideal.span (Inser …
  -/
  rw [← IsBezout.span_gcd, IsBezout.span_gcd_eq_span_gcd]
  /-
    🎉 no goals
  -/


theorem gcd_dvd_iff_exists (a b : R) {z} : gcd a b ∣ z ↔ ∃ x y, z = a * x + b * y := by
  simp_rw [mul_comm a, mul_comm b, @eq_comm _ z, ← Ideal.mem_span_pair, ← span_gcd,
    Ideal.mem_span_singleton]


/-- **Bézout's lemma** -/
theorem exists_gcd_eq_mul_add_mul (a b : R) : ∃ x y, gcd a b = a * x + b * y := by
  /-
    R : Type u
    inst✝³ : CommRing R
    inst✝² : IsBezout R
    inst✝¹ : IsDomain R
    inst✝ : GCDMonoid R
    a b : R
    ⊢ Exists fun x => Exists fun y => Eq (GCDMonoid.gcd a b) (HAdd.hAdd (HMul.hMul …
  -/
  rw [← gcd_dvd_iff_exists]
  /-
    🎉 no goals
  -/


theorem gcd_isUnit_iff (x y : R) : IsUnit (gcd x y) ↔ IsCoprime x y := by
  /-
    R : Type u
    inst✝³ : CommRing R
    inst✝² : IsBezout R
    inst✝¹ : IsDomain R
    inst✝ : GCDMonoid R
    x y : R
    ⊢ Iff (IsUnit (GCDMonoid.gcd x y)) (IsCoprime x y)
  -/
  rw [IsCoprime, ← Ideal.mem_span_pair, ← span_gcd, ← span_singleton_eq_top, eq_top_iff_one]
  /-
    🎉 no goals
  -/


theorem Prime.coprime_iff_not_dvd {p n : R} (hp : Prime p) : IsCoprime p n ↔ ¬p ∣ n :=
  hp.irreducible.coprime_iff_not_dvd


theorem exists_associated_pow_of_mul_eq_pow' {a b c : R} (hab : IsCoprime a b) {k : ℕ}
    (h : a * b = c ^ k) : ∃ d : R, Associated (d ^ k) a := by
  classical
  letI := IsBezout.toGCDDomain R
  exact exists_associated_pow_of_mul_eq_pow ((gcd_isUnit_iff _ _).mpr hab) h


theorem exists_associated_pow_of_associated_pow_mul {a b c : R} (hab : IsCoprime a b) {k : ℕ}
    (h : Associated (c ^ k) (a * b)) : ∃ d : R, Associated (d ^ k) a := by
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsBezout R
    inst✝ : IsDomain R
    a b c : R
    hab : IsCoprime a b
    k : Nat
    h : Associated (HPow.hPow c k) (HMul.hMul a b)
    ⊢ Exists fun d => Associated (HPow.hPow d k) a
  -/
  obtain ⟨u, hu⟩ := h.symm
  exact exists_associated_pow_of_mul_eq_pow'
    ((isCoprime_mul_unit_right_right u.isUnit a b).mpr hab) <| mul_assoc a _ _ ▸ hu


theorem isCoprime_of_irreducible_dvd {x y : R} (nonzero : ¬(x = 0 ∧ y = 0))
    (H : ∀ z : R, Irreducible z → z ∣ x → ¬z ∣ y) : IsCoprime x y :=
  (WfDvdMonoid.isRelPrime_of_no_irreducible_factors nonzero H).isCoprime


theorem isCoprime_of_prime_dvd {x y : R} (nonzero : ¬(x = 0 ∧ y = 0))
    (H : ∀ z : R, Prime z → z ∣ x → ¬z ∣ y) : IsCoprime x y :=
  isCoprime_of_irreducible_dvd nonzero fun z zi ↦ H z zi.prime


/-- `nonPrincipals R` is the set of all ideals of `R` that are not principal ideals. -/
def nonPrincipals :=
  { I : Ideal R | ¬I.IsPrincipal }


theorem nonPrincipals_def {I : Ideal R} : I ∈ nonPrincipals R ↔ ¬I.IsPrincipal :=
  Iff.rfl


theorem nonPrincipals_eq_empty_iff : nonPrincipals R = ∅ ↔ IsPrincipalIdealRing R := by
  /-
    R : Type u
    inst✝ : CommRing R
    ⊢ Iff (Eq (nonPrincipals R) EmptyCollection.emptyCollection) (IsPrincipalIdeal …
  -/
  simp [Set.eq_empty_iff_forall_not_mem, isPrincipalIdealRing_iff, nonPrincipals_def]
  /-
    🎉 no goals
  -/


/-- Any chain in the set of non-principal ideals has an upper bound which is non-principal.
(Namely, the union of the chain is such an upper bound.)
-/
theorem nonPrincipals_zorn (c : Set (Ideal R)) (hs : c ⊆ nonPrincipals R)
    (hchain : IsChain (· ≤ ·) c) {K : Ideal R} (hKmem : K ∈ c) :
    ∃ I ∈ nonPrincipals R, ∀ J ∈ c, J ≤ I := by
  /-
    R : Type u
    inst✝ : CommRing R
    c : Set (Ideal R)
    hs : HasSubset.Subset c (nonPrincipals R)
    hchain : IsChain (fun x1 x2 => LE.le x1 x2) c
    K : Ideal R
    hKmem : Membership.mem c K
    ⊢ Exists fun I => And (Membership.mem (nonPrincipals R) I) (∀ (J : Ideal R), M …
  -/
  refine ⟨sSup c, ?_, fun J hJ => le_sSup hJ⟩
  /-
    R : Type u
    inst✝ : CommRing R
    c : Set (Ideal R)
    hs : HasSubset.Subset c (nonPrincipals R)
    hchain : IsChain (fun x1 x2 => LE.le x1 x2) c
    K : Ideal R
    hKmem : Membership.mem c K
    ⊢ Membership.mem (nonPrincipals R) (SupSet.sSup c)
  -/
  rintro ⟨x, hx⟩
  /-
    case mk.intro
    R : Type u
    inst✝ : CommRing R
    c : Set (Ideal R)
    hs : HasSubset.Subset c (nonPrincipals R)
    hchain : IsChain (fun x1 x2 => LE.le x1 x2) c
    K : Ideal R
    hKmem : Membership.mem c K
    x : R
    hx : Eq (SupSet.sSup c) (Submodule.span R (Singleton.singleton x))
    ⊢ False
  -/
  have hxmem : x ∈ sSup c := hx.symm ▸ Submodule.mem_span_singleton_self x
  /-
    case mk.intro
    R : Type u
    inst✝ : CommRing R
    c : Set (Ideal R)
    hs : HasSubset.Subset c (nonPrincipals R)
    hchain : IsChain (fun x1 x2 => LE.le x1 x2) c
    K : Ideal R
    hKmem : Membership.mem c K
    x : R
    hx : Eq (SupSet.sSup c) (Submodule.span R (Singleton.singleton x))
    hxmem : Membership.mem (SupSet.sSup c) x
    ⊢ False
  -/
  obtain ⟨J, hJc, hxJ⟩ := (Submodule.mem_sSup_of_directed ⟨K, hKmem⟩ hchain.directedOn).1 hxmem
  /-
    case mk.intro.intro.intro
    R : Type u
    inst✝ : CommRing R
    c : Set (Ideal R)
    hs : HasSubset.Subset c (nonPrincipals R)
    hchain : IsChain (fun x1 x2 => LE.le x1 x2) c
    K : Ideal R
    hKmem : Membership.mem c K
    x : R
    hx : Eq (SupSet.sSup c) (Submodule.span R (Singleton.singleton x))
    hxmem : Membership.mem (SupSet.sSup c) x
    J : Submodule R R
    hJc : Membership.mem c J
    hxJ : Membership.mem J x
    ⊢ False
  -/
  have hsSupJ : sSup c = J := le_antisymm (by simp [hx, Ideal.span_le, hxJ]) (le_sSup hJc)
  /-
    case mk.intro.intro.intro
    R : Type u
    inst✝ : CommRing R
    c : Set (Ideal R)
    hs : HasSubset.Subset c (nonPrincipals R)
    hchain : IsChain (fun x1 x2 => LE.le x1 x2) c
    K : Ideal R
    hKmem : Membership.mem c K
    x : R
    hx : Eq (SupSet.sSup c) (Submodule.span R (Singleton.singleton x))
    hxmem : Membership.mem (SupSet.sSup c) x
    J : Submodule R R
    hJc : Membership.mem c J
    hxJ : Membership.mem J x
    hsSupJ : Eq (SupSet.sSup c) J
    ⊢ False
  -/
  specialize hs hJc
  /-
    case mk.intro.intro.intro
    R : Type u
    inst✝ : CommRing R
    c : Set (Ideal R)
    hchain : IsChain (fun x1 x2 => LE.le x1 x2) c
    K : Ideal R
    hKmem : Membership.mem c K
    x : R
    hx : Eq (SupSet.sSup c) (Submodule.span R (Singleton.singleton x))
    hxmem : Membership.mem (SupSet.sSup c) x
    J : Submodule R R
    hJc : Membership.mem c J
    hxJ : Membership.mem J x
    hsSupJ : Eq (SupSet.sSup c) J
    hs : Membership.mem (nonPrincipals R) J
    ⊢ False
  -/
  rw [← hsSupJ, hx, nonPrincipals_def] at hs
  /-
    case mk.intro.intro.intro
    R : Type u
    inst✝ : CommRing R
    c : Set (Ideal R)
    hchain : IsChain (fun x1 x2 => LE.le x1 x2) c
    K : Ideal R
    hKmem : Membership.mem c K
    x : R
    hx : Eq (SupSet.sSup c) (Submodule.span R (Singleton.singleton x))
    hxmem : Membership.mem (SupSet.sSup c) x
    J : Submodule R R
    hJc : Membership.mem c J
    hxJ : Membership.mem J x
    hsSupJ : Eq (SupSet.sSup c) J
    hs : Not (Submodule.span R (Singleton.singleton x)).IsPrincipal
    ⊢ False
  -/
  exact hs ⟨⟨x, rfl⟩⟩
  /-
    🎉 no goals
  -/


