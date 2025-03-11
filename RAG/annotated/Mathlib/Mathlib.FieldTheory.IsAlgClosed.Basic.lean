/-- Typeclass for algebraically closed fields.

To show `Polynomial.Splits p f` for an arbitrary ring homomorphism `f`,
see `IsAlgClosed.splits_codomain` and `IsAlgClosed.splits_domain`.
-/
@[stacks 09GR "The definition of `IsAlgClosed` in mathlib is 09GR (4)"]
class IsAlgClosed : Prop where
  splits : ∀ p : k[X], p.Splits <| RingHom.id k


/-- Every polynomial splits in the field extension `f : K →+* k` if `k` is algebraically closed.

See also `IsAlgClosed.splits_domain` for the case where `K` is algebraically closed.
-/
theorem IsAlgClosed.splits_codomain {k K : Type*} [Field k] [IsAlgClosed k] [Field K] {f : K →+* k}
                                  /-
                                    k : Type u_1
                                    K : Type u_2
                                    inst✝² : Field k
                                    inst✝¹ : IsAlgClosed k
                                    inst✝ : Field K
                                    f : RingHom K k
                                    p : Polynomial K
                                    ⊢ Polynomial.Splits f p
                                  -/
    (p : K[X]) : p.Splits f := by convert IsAlgClosed.splits (p.map f); simp [splits_map_iff]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


/-- Every polynomial splits in the field extension `f : K →+* k` if `K` is algebraically closed.

See also `IsAlgClosed.splits_codomain` for the case where `k` is algebraically closed.
-/
theorem IsAlgClosed.splits_domain {k K : Type*} [Field k] [IsAlgClosed k] [Field K] {f : k →+* K}
    (p : k[X]) : p.Splits f :=
  Polynomial.splits_of_splits_id _ <| IsAlgClosed.splits _


/--
If `k` is algebraically closed, then every nonconstant polynomial has a root.
-/
@[stacks 09GR "(4) ⟹ (3)"]
theorem exists_root [IsAlgClosed k] (p : k[X]) (hp : p.degree ≠ 0) : ∃ x, IsRoot p x :=
  exists_root_of_splits _ (IsAlgClosed.splits p) hp


theorem exists_pow_nat_eq [IsAlgClosed k] (x : k) {n : ℕ} (hn : 0 < n) : ∃ z, z ^ n = x := by
  have : degree (X ^ n - C x) ≠ 0 := by
    rw [degree_X_pow_sub_C hn x]
    exact ne_of_gt (WithBot.coe_lt_coe.2 hn)
  /-
    k : Type u
    inst✝¹ : Field k
    inst✝ : IsAlgClosed k
    x : k
    n : Nat
    hn : LT.lt 0 n
    this : Ne (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C x)).degree 0
    ⊢ Exists fun z => Eq (HPow.hPow z n) x
  -/
  obtain ⟨z, hz⟩ := exists_root (X ^ n - C x) this
  /-
    case intro
    k : Type u
    inst✝¹ : Field k
    inst✝ : IsAlgClosed k
    x : k
    n : Nat
    hn : LT.lt 0 n
    this : Ne (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C x)).degree 0
    z : k
    hz : (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C x)).IsRoot z
    ⊢ Exists fun z => Eq (HPow.hPow z n) x
  -/
  use z
  /-
    case h
    k : Type u
    inst✝¹ : Field k
    inst✝ : IsAlgClosed k
    x : k
    n : Nat
    hn : LT.lt 0 n
    this : Ne (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C x)).degree 0
    z : k
    hz : (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C x)).IsRoot z
    ⊢ Eq (HPow.hPow z n) x
  -/
  simp only [eval_C, eval_X, eval_pow, eval_sub, IsRoot.def] at hz
  /-
    case h
    k : Type u
    inst✝¹ : Field k
    inst✝ : IsAlgClosed k
    x : k
    n : Nat
    hn : LT.lt 0 n
    this : Ne (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C x)).degree 0
    z : k
    hz : Eq (HSub.hSub (HPow.hPow z n) x) 0
    ⊢ Eq (HPow.hPow z n) x
  -/
  exact sub_eq_zero.1 hz
  /-
    🎉 no goals
  -/


theorem exists_eq_mul_self [IsAlgClosed k] (x : k) : ∃ z, x = z * z := by
  /-
    k : Type u
    inst✝¹ : Field k
    inst✝ : IsAlgClosed k
    x : k
    ⊢ Exists fun z => Eq x (HMul.hMul z z)
  -/
  rcases exists_pow_nat_eq x zero_lt_two with ⟨z, rfl⟩
  /-
    case intro
    k : Type u
    inst✝¹ : Field k
    inst✝ : IsAlgClosed k
    z : k
    ⊢ Exists fun z_1 => Eq (HPow.hPow z 2) (HMul.hMul z_1 z_1)
  -/
  exact ⟨z, sq z⟩
  /-
    🎉 no goals
  -/


theorem roots_eq_zero_iff [IsAlgClosed k] {p : k[X]} :
    p.roots = 0 ↔ p = Polynomial.C (p.coeff 0) := by
  /-
    k : Type u
    inst✝¹ : Field k
    inst✝ : IsAlgClosed k
    p : Polynomial k
    ⊢ Iff (Eq p.roots 0) (Eq p (Polynomial.C (p.coeff 0)))
  -/
  refine ⟨fun h => ?_, fun hp => by rw [hp, roots_C]⟩
  /-
    k : Type u
    inst✝¹ : Field k
    inst✝ : IsAlgClosed k
    p : Polynomial k
    h : Eq p.roots 0
    ⊢ Eq p (Polynomial.C (p.coeff 0))
  -/
  rcases le_or_lt (degree p) 0 with hd | hd
    /-
      case inl
      k : Type u
      inst✝¹ : Field k
      inst✝ : IsAlgClosed k
      p : Polynomial k
      h : Eq p.roots 0
      hd : LE.le p.degree 0
      ⊢ Eq p (Polynomial.C (p.coeff 0))
    -/
  · exact eq_C_of_degree_le_zero hd
    /-
      🎉 no goals
    -/
    /-
      case inr
      k : Type u
      inst✝¹ : Field k
      inst✝ : IsAlgClosed k
      p : Polynomial k
      h : Eq p.roots 0
      hd : LT.lt 0 p.degree
      ⊢ Eq p (Polynomial.C (p.coeff 0))
    -/
  · obtain ⟨z, hz⟩ := IsAlgClosed.exists_root p hd.ne'
    /-
      case inr.intro
      k : Type u
      inst✝¹ : Field k
      inst✝ : IsAlgClosed k
      p : Polynomial k
      h : Eq p.roots 0
      hd : LT.lt 0 p.degree
      z : k
      hz : p.IsRoot z
      ⊢ Eq p (Polynomial.C (p.coeff 0))
    -/
    rw [← mem_roots (ne_zero_of_degree_gt hd), h] at hz
    /-
      case inr.intro
      k : Type u
      inst✝¹ : Field k
      inst✝ : IsAlgClosed k
      p : Polynomial k
      h : Eq p.roots 0
      hd : LT.lt 0 p.degree
      z : k
      hz : Membership.mem 0 z
      ⊢ Eq p (Polynomial.C (p.coeff 0))
    -/
    simp at hz
    /-
      🎉 no goals
    -/


theorem exists_eval₂_eq_zero_of_injective {R : Type*} [Ring R] [IsAlgClosed k] (f : R →+* k)
    (hf : Function.Injective f) (p : R[X]) (hp : p.degree ≠ 0) : ∃ x, p.eval₂ f x = 0 :=
                                           /-
                                             k : Type u
                                             inst✝² : Field k
                                             R : Type u_1
                                             inst✝¹ : Ring R
                                             inst✝ : IsAlgClosed k
                                             f : RingHom R k
                                             hf : Function.Injective ⇑f
                                             p : Polynomial R
                                             hp : Ne p.degree 0
                                             ⊢ Ne (Polynomial.map f p).degree 0
                                           -/
  let ⟨x, hx⟩ := exists_root (p.map f) (by rwa [degree_map_eq_of_injective hf])
                                           /-
                                             🎉 no goals
                                           -/
         /-
           k : Type u
           inst✝² : Field k
           R : Type u_1
           inst✝¹ : Ring R
           inst✝ : IsAlgClosed k
           f : RingHom R k
           hf : Function.Injective ⇑f
           p : Polynomial R
           hp : Ne p.degree 0
           x : k
           hx : (Polynomial.map f p).IsRoot x
           ⊢ Eq (Polynomial.eval₂ f x p) 0
         -/
  ⟨x, by rwa [eval₂_eq_eval_map, ← IsRoot]⟩
         /-
           🎉 no goals
         -/


theorem exists_eval₂_eq_zero {R : Type*} [Field R] [IsAlgClosed k] (f : R →+* k) (p : R[X])
    (hp : p.degree ≠ 0) : ∃ x, p.eval₂ f x = 0 :=
  exists_eval₂_eq_zero_of_injective f f.injective p hp


theorem exists_aeval_eq_zero_of_injective {R : Type*} [CommRing R] [IsAlgClosed k] [Algebra R k]
    (hinj : Function.Injective (algebraMap R k)) (p : R[X]) (hp : p.degree ≠ 0) :
    ∃ x : k, aeval x p = 0 :=
  exists_eval₂_eq_zero_of_injective (algebraMap R k) hinj p hp


theorem exists_aeval_eq_zero {R : Type*} [Field R] [IsAlgClosed k] [Algebra R k] (p : R[X])
    (hp : p.degree ≠ 0) : ∃ x : k, aeval x p = 0 :=
  exists_eval₂_eq_zero (algebraMap R k) p hp



/--
If every nonconstant polynomial over `k` has a root, then `k` is algebraically closed.
-/
@[stacks 09GR "(3) ⟹ (4)"]
theorem of_exists_root (H : ∀ p : k[X], p.Monic → Irreducible p → ∃ x, p.eval x = 0) :
    IsAlgClosed k := by
  /-
    k : Type u
    inst✝ : Field k
    H : ∀ (p : Polynomial k), p.Monic → Irreducible p → Exists fun x => Eq (Polyno …
    ⊢ IsAlgClosed k
  -/
  refine ⟨fun p ↦ Or.inr ?_⟩
  /-
    k : Type u
    inst✝ : Field k
    H : ∀ (p : Polynomial k), p.Monic → Irreducible p → Exists fun x => Eq (Polyno …
    p : Polynomial k
    ⊢ ∀ {g : Polynomial k}, Irreducible g → Dvd.dvd g (Polynomial.map (RingHom.id  …
  -/
  intro q hq _
  have : Irreducible (q * C (leadingCoeff q)⁻¹) := by
    classical
    rw [← coe_normUnit_of_ne_zero hq.ne_zero]
    exact (associated_normalize _).irreducible hq
  /-
    k : Type u
    inst✝ : Field k
    H : ∀ (p : Polynomial k), p.Monic → Irreducible p → Exists fun x => Eq (Polyno …
    p q : Polynomial k
    hq : Irreducible q
    a✝ : Dvd.dvd q (Polynomial.map (RingHom.id k) p)
    this : Irreducible (HMul.hMul q (Polynomial.C (Inv.inv q.leadingCoeff)))
    ⊢ Eq q.degree 1
  -/
  obtain ⟨x, hx⟩ := H (q * C (leadingCoeff q)⁻¹) (monic_mul_leadingCoeff_inv hq.ne_zero) this
  /-
    case intro
    k : Type u
    inst✝ : Field k
    H : ∀ (p : Polynomial k), p.Monic → Irreducible p → Exists fun x => Eq (Polyno …
    p q : Polynomial k
    hq : Irreducible q
    a✝ : Dvd.dvd q (Polynomial.map (RingHom.id k) p)
    this : Irreducible (HMul.hMul q (Polynomial.C (Inv.inv q.leadingCoeff)))
    x : k
    hx : Eq (Polynomial.eval x (HMul.hMul q (Polynomial.C (Inv.inv q.leadingCoeff) …
    ⊢ Eq q.degree 1
  -/
  exact degree_mul_leadingCoeff_inv q hq.ne_zero ▸ degree_eq_one_of_irreducible_of_root this hx
  /-
    🎉 no goals
  -/


theorem of_ringEquiv (k' : Type u) [Field k'] (e : k ≃+* k')
    [IsAlgClosed k] : IsAlgClosed k' := by
  /-
    k : Type u
    inst✝² : Field k
    k' : Type u
    inst✝¹ : Field k'
    e : RingEquiv k k'
    inst✝ : IsAlgClosed k
    ⊢ IsAlgClosed k'
  -/
  apply IsAlgClosed.of_exists_root
  /-
    case H
    k : Type u
    inst✝² : Field k
    k' : Type u
    inst✝¹ : Field k'
    e : RingEquiv k k'
    inst✝ : IsAlgClosed k
    ⊢ ∀ (p : Polynomial k'), p.Monic → Irreducible p → Exists fun x => Eq (Polynom …
  -/
  intro p hmp hp
  have hpe : degree (p.map e.symm.toRingHom) ≠ 0 := by
    rw [degree_map]
    exact ne_of_gt (degree_pos_of_irreducible hp)
  /-
    case H
    k : Type u
    inst✝² : Field k
    k' : Type u
    inst✝¹ : Field k'
    e : RingEquiv k k'
    inst✝ : IsAlgClosed k
    p : Polynomial k'
    hmp : p.Monic
    hp : Irreducible p
    hpe : Ne (Polynomial.map e.symm.toRingHom p).degree 0
    ⊢ Exists fun x => Eq (Polynomial.eval x p) 0
  -/
  rcases IsAlgClosed.exists_root (k := k) (p.map e.symm) hpe with ⟨x, hx⟩
  /-
    case H.intro
    k : Type u
    inst✝² : Field k
    k' : Type u
    inst✝¹ : Field k'
    e : RingEquiv k k'
    inst✝ : IsAlgClosed k
    p : Polynomial k'
    hmp : p.Monic
    hp : Irreducible p
    hpe : Ne (Polynomial.map e.symm.toRingHom p).degree 0
    x : k
    hx : (Polynomial.map (↑e.symm) p).IsRoot x
    ⊢ Exists fun x => Eq (Polynomial.eval x p) 0
  -/
  use e x
  /-
    case h
    k : Type u
    inst✝² : Field k
    k' : Type u
    inst✝¹ : Field k'
    e : RingEquiv k k'
    inst✝ : IsAlgClosed k
    p : Polynomial k'
    hmp : p.Monic
    hp : Irreducible p
    hpe : Ne (Polynomial.map e.symm.toRingHom p).degree 0
    x : k
    hx : (Polynomial.map (↑e.symm) p).IsRoot x
    ⊢ Eq (Polynomial.eval (e x) p) 0
  -/
  rw [IsRoot] at hx
  /-
    case h
    k : Type u
    inst✝² : Field k
    k' : Type u
    inst✝¹ : Field k'
    e : RingEquiv k k'
    inst✝ : IsAlgClosed k
    p : Polynomial k'
    hmp : p.Monic
    hp : Irreducible p
    hpe : Ne (Polynomial.map e.symm.toRingHom p).degree 0
    x : k
    hx : Eq (Polynomial.eval x (Polynomial.map (↑e.symm) p)) 0
    ⊢ Eq (Polynomial.eval (e x) p) 0
  -/
  apply e.symm.injective
  /-
    case h.a
    k : Type u
    inst✝² : Field k
    k' : Type u
    inst✝¹ : Field k'
    e : RingEquiv k k'
    inst✝ : IsAlgClosed k
    p : Polynomial k'
    hmp : p.Monic
    hp : Irreducible p
    hpe : Ne (Polynomial.map e.symm.toRingHom p).degree 0
    x : k
    hx : Eq (Polynomial.eval x (Polynomial.map (↑e.symm) p)) 0
    ⊢ Eq (e.symm (Polynomial.eval (e x) p)) (e.symm 0)
  -/
  rw [map_zero, ← hx]
  /-
    case h.a
    k : Type u
    inst✝² : Field k
    k' : Type u
    inst✝¹ : Field k'
    e : RingEquiv k k'
    inst✝ : IsAlgClosed k
    p : Polynomial k'
    hmp : p.Monic
    hp : Irreducible p
    hpe : Ne (Polynomial.map e.symm.toRingHom p).degree 0
    x : k
    hx : Eq (Polynomial.eval x (Polynomial.map (↑e.symm) p)) 0
    ⊢ Eq (e.symm (Polynomial.eval (e x) p)) (Polynomial.eval x (Polynomial.map (↑e …
  -/
  clear hx hpe hp hmp
  /-
    case h.a
    k : Type u
    inst✝² : Field k
    k' : Type u
    inst✝¹ : Field k'
    e : RingEquiv k k'
    inst✝ : IsAlgClosed k
    p : Polynomial k'
    x : k
    ⊢ Eq (e.symm (Polynomial.eval (e x) p)) (Polynomial.eval x (Polynomial.map (↑e …
  -/
                                                /-
                                                  🎉 no goals
                                                -/
                                                /-
                                                  🎉 no goals
                                                -/
  induction p using Polynomial.induction_on <;> simp_all
                                                /-
                                                  🎉 no goals
                                                -/


/--
If `k` is algebraically closed, then every irreducible polynomial over `k` is linear.
-/
@[stacks 09GR "(4) ⟹ (2)"]
theorem degree_eq_one_of_irreducible [IsAlgClosed k] {p : k[X]} (hp : Irreducible p) :
    p.degree = 1 :=
  degree_eq_one_of_irreducible_of_splits hp (IsAlgClosed.splits_codomain _)


theorem algebraMap_surjective_of_isIntegral {k K : Type*} [Field k] [Ring K] [IsDomain K]
    [hk : IsAlgClosed k] [Algebra k K] [Algebra.IsIntegral k K] :
    Function.Surjective (algebraMap k K) := by
  /-
    k : Type u_1
    K : Type u_2
    inst✝⁴ : Field k
    inst✝³ : Ring K
    inst✝² : IsDomain K
    hk : IsAlgClosed k
    inst✝¹ : Algebra k K
    inst✝ : Algebra.IsIntegral k K
    ⊢ Function.Surjective ⇑(algebraMap k K)
  -/
  refine fun x => ⟨-(minpoly k x).coeff 0, ?_⟩
  /-
    k : Type u_1
    K : Type u_2
    inst✝⁴ : Field k
    inst✝³ : Ring K
    inst✝² : IsDomain K
    hk : IsAlgClosed k
    inst✝¹ : Algebra k K
    inst✝ : Algebra.IsIntegral k K
    x : K
    ⊢ Eq ((algebraMap k K) (Neg.neg ((minpoly k x).coeff 0))) x
  -/
  have hq : (minpoly k x).leadingCoeff = 1 := minpoly.monic (Algebra.IsIntegral.isIntegral x)
  have h : (minpoly k x).degree = 1 := degree_eq_one_of_irreducible k (minpoly.irreducible
    (Algebra.IsIntegral.isIntegral x))
  /-
    k : Type u_1
    K : Type u_2
    inst✝⁴ : Field k
    inst✝³ : Ring K
    inst✝² : IsDomain K
    hk : IsAlgClosed k
    inst✝¹ : Algebra k K
    inst✝ : Algebra.IsIntegral k K
    x : K
    hq : Eq (minpoly k x).leadingCoeff 1
    h : Eq (minpoly k x).degree 1
    ⊢ Eq ((algebraMap k K) (Neg.neg ((minpoly k x).coeff 0))) x
  -/
  have : aeval x (minpoly k x) = 0 := minpoly.aeval k x
  rw [eq_X_add_C_of_degree_eq_one h, hq, C_1, one_mul, aeval_add, aeval_X, aeval_C,
    add_eq_zero_iff_eq_neg] at this
  /-
    k : Type u_1
    K : Type u_2
    inst✝⁴ : Field k
    inst✝³ : Ring K
    inst✝² : IsDomain K
    hk : IsAlgClosed k
    inst✝¹ : Algebra k K
    inst✝ : Algebra.IsIntegral k K
    x : K
    hq : Eq (minpoly k x).leadingCoeff 1
    h : Eq (minpoly k x).degree 1
    this : Eq x (Neg.neg ((algebraMap k K) ((minpoly k x).coeff 0)))
    ⊢ Eq ((algebraMap k K) (Neg.neg ((minpoly k x).coeff 0))) x
  -/
  exact (RingHom.map_neg (algebraMap k K) ((minpoly k x).coeff 0)).symm ▸ this.symm
  /-
    🎉 no goals
  -/


theorem algebraMap_surjective_of_isIntegral' {k K : Type*} [Field k] [CommRing K] [IsDomain K]
    [IsAlgClosed k] (f : k →+* K) (hf : f.IsIntegral) : Function.Surjective f :=
  let _ : Algebra k K := f.toAlgebra
  have : Algebra.IsIntegral k K := ⟨hf⟩
  algebraMap_surjective_of_isIntegral


/--
Deprecated: `algebraMap_surjective_of_isIntegral` is identical apart from the `IsIntegral` argument,
which can be found by instance synthesis
-/
@[deprecated algebraMap_surjective_of_isIntegral (since := "2024-05-08")]
theorem algebraMap_surjective_of_isAlgebraic {k K : Type*} [Field k] [Ring K] [IsDomain K]
    [IsAlgClosed k] [Algebra k K] [Algebra.IsAlgebraic k K] :
    Function.Surjective (algebraMap k K) :=
  algebraMap_surjective_of_isIntegral


/-- If `k` is algebraically closed, `K / k` is a field extension, `L / k` is an intermediate field
which is algebraic, then `L` is equal to `k`. A corollary of
`IsAlgClosed.algebraMap_surjective_of_isAlgebraic`. -/
@[stacks 09GQ "The result is the definition of algebraically closedness in Stacks Project. \
This statement is 09GR (4) ⟹ (1)."]
theorem IntermediateField.eq_bot_of_isAlgClosed_of_isAlgebraic {k K : Type*} [Field k] [Field K]
    [IsAlgClosed k] [Algebra k K] (L : IntermediateField k K) [Algebra.IsAlgebraic k L] :
    L = ⊥ := bot_unique fun x hx ↦ by
  /-
    k : Type u_1
    K : Type u_2
    inst✝⁴ : Field k
    inst✝³ : Field K
    inst✝² : IsAlgClosed k
    inst✝¹ : Algebra k K
    L : IntermediateField k K
    inst✝ : Algebra.IsAlgebraic k (Subtype fun x => Membership.mem L x)
    x : K
    hx : Membership.mem L x
    ⊢ Membership.mem Bot.bot x
  -/
  obtain ⟨y, hy⟩ := IsAlgClosed.algebraMap_surjective_of_isIntegral (k := k) (⟨x, hx⟩ : L)
  /-
    case intro
    k : Type u_1
    K : Type u_2
    inst✝⁴ : Field k
    inst✝³ : Field K
    inst✝² : IsAlgClosed k
    inst✝¹ : Algebra k K
    L : IntermediateField k K
    inst✝ : Algebra.IsAlgebraic k (Subtype fun x => Membership.mem L x)
    x : K
    hx : Membership.mem L x
    y : k
    hy : Eq ((algebraMap k (Subtype fun x => Membership.mem L x)) y) ⟨x, hx⟩
    ⊢ Membership.mem Bot.bot x
  -/
  exact ⟨y, congr_arg (algebraMap L K) hy⟩
  /-
    🎉 no goals
  -/


lemma Polynomial.isCoprime_iff_aeval_ne_zero_of_isAlgClosed (K : Type v) [Field K] [IsAlgClosed K]
    [Algebra k K] (p q : k[X]) : IsCoprime p q ↔ ∀ a : K, aeval a p ≠ 0 ∨ aeval a q ≠ 0 := by
  /-
    k : Type u
    inst✝³ : Field k
    K : Type v
    inst✝² : Field K
    inst✝¹ : IsAlgClosed K
    inst✝ : Algebra k K
    p q : Polynomial k
    ⊢ Iff (IsCoprime p q) (∀ (a : K), Or (Ne ((Polynomial.aeval a) p) 0) (Ne ((Pol …
  -/
  refine ⟨fun h => aeval_ne_zero_of_isCoprime h, fun h => isCoprime_of_dvd _ _ ?_ fun x hu h0 => ?_⟩
    /-
      case refine_1
      k : Type u
      inst✝³ : Field k
      K : Type v
      inst✝² : Field K
      inst✝¹ : IsAlgClosed K
      inst✝ : Algebra k K
      p q : Polynomial k
      h : ∀ (a : K), Or (Ne ((Polynomial.aeval a) p) 0) (Ne ((Polynomial.aeval a) q) …
      ⊢ Not (And (Eq p 0) (Eq q 0))
    -/
  · replace h := h 0
    /-
      case refine_1
      k : Type u
      inst✝³ : Field k
      K : Type v
      inst✝² : Field K
      inst✝¹ : IsAlgClosed K
      inst✝ : Algebra k K
      p q : Polynomial k
      h : Or (Ne ((Polynomial.aeval 0) p) 0) (Ne ((Polynomial.aeval 0) q) 0)
      ⊢ Not (And (Eq p 0) (Eq q 0))
    -/
    contrapose! h
    /-
      case refine_1
      k : Type u
      inst✝³ : Field k
      K : Type v
      inst✝² : Field K
      inst✝¹ : IsAlgClosed K
      inst✝ : Algebra k K
      p q : Polynomial k
      h : And (Eq p 0) (Eq q 0)
      ⊢ And (Eq ((Polynomial.aeval 0) p) 0) (Eq ((Polynomial.aeval 0) q) 0)
    -/
    rw [h.left, h.right, map_zero, and_self]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      k : Type u
      inst✝³ : Field k
      K : Type v
      inst✝² : Field K
      inst✝¹ : IsAlgClosed K
      inst✝ : Algebra k K
      p q : Polynomial k
      h : ∀ (a : K), Or (Ne ((Polynomial.aeval a) p) 0) (Ne ((Polynomial.aeval a) q) …
      x : Polynomial k
      hu : Membership.mem (nonunits (Polynomial k)) x
      h0 : Ne x 0
      ⊢ Dvd.dvd x p → Not (Dvd.dvd x q)
    -/
  · rintro ⟨_, rfl⟩ ⟨_, rfl⟩
    obtain ⟨a, ha : _ = _⟩ := IsAlgClosed.exists_root (x.map <| algebraMap k K) <| by
      simpa only [degree_map] using (ne_of_lt <| degree_pos_of_ne_zero_of_nonunit h0 hu).symm
    /-
      case refine_2.intro.intro.intro
      k : Type u
      inst✝³ : Field k
      K : Type v
      inst✝² : Field K
      inst✝¹ : IsAlgClosed K
      inst✝ : Algebra k K
      x : Polynomial k
      hu : Membership.mem (nonunits (Polynomial k)) x
      h0 : Ne x 0
      w✝¹ w✝ : Polynomial k
      h : ∀ (a : K), Or (Ne ((Polynomial.aeval a) (HMul.hMul x w✝¹)) 0) (Ne ((Polyno …
      a : K
      ha : Eq (Polynomial.eval a (Polynomial.map (algebraMap k K) x)) 0
      ⊢ False
    -/
    exact not_and_or.mpr (h a) (by simp_rw [map_mul, ← eval_map_algebraMap, ha, zero_mul, true_and])
    /-
      🎉 no goals
    -/


/-- Typeclass for an extension being an algebraic closure. -/
@[stacks 09GS]
class IsAlgClosure (R : Type u) (K : Type v) [CommRing R] [Field K] [Algebra R K]
    [NoZeroSMulDivisors R K] : Prop where
  isAlgClosed : IsAlgClosed K
  isAlgebraic : Algebra.IsAlgebraic R K


@[deprecated (since := "2024-08-31")] alias IsAlgClosure.alg_closed := IsAlgClosure.isAlgClosed

@[deprecated (since := "2024-08-31")] alias IsAlgClosure.algebraic := IsAlgClosure.isAlgebraic


theorem isAlgClosure_iff (K : Type v) [Field K] [Algebra k K] :
    IsAlgClosure k K ↔ IsAlgClosed K ∧ Algebra.IsAlgebraic k K :=
  ⟨fun h => ⟨h.1, h.2⟩, fun h => ⟨h.1, h.2⟩⟩


instance (priority := 100) IsAlgClosure.normal (R K : Type*) [Field R] [Field K] [Algebra R K]
    [IsAlgClosure R K] : Normal R K where
  toIsAlgebraic := IsAlgClosure.isAlgebraic
  splits' _ := @IsAlgClosed.splits_codomain _ _ _ (IsAlgClosure.isAlgClosed R) _ _ _


instance (priority := 100) IsAlgClosure.separable (R K : Type*) [Field R] [Field K] [Algebra R K]
    [IsAlgClosure R K] [CharZero R] : Algebra.IsSeparable R K :=
  ⟨fun _ => (minpoly.irreducible (Algebra.IsIntegral.isIntegral _)).separable⟩


instance IsAlgClosed.instIsAlgClosure (F : Type*) [Field F] [IsAlgClosed F] : IsAlgClosure F F where
  isAlgClosed := ‹_›
  isAlgebraic := .of_finite F F


theorem IsAlgClosure.of_splits {R K} [CommRing R] [IsDomain R] [Field K] [Algebra R K]
    [Algebra.IsIntegral R K] [NoZeroSMulDivisors R K]
    (h : ∀ p : R[X], p.Monic → Irreducible p → p.Splits (algebraMap R K)) : IsAlgClosure R K where
  isAlgebraic := inferInstance
  isAlgClosed := .of_exists_root _ fun _p _ p_irred ↦
    have ⟨g, monic, irred, dvd⟩ := p_irred.exists_dvd_monic_irreducible_of_isIntegral (K := R)
    exists_root_of_splits _ (splits_of_splits_of_dvd _ (map_monic_ne_zero monic)
      ((splits_id_iff_splits _).mpr <| h g monic irred) dvd) <|
        degree_ne_of_natDegree_ne p_irred.natDegree_pos.ne'


/-- If E/L/K is a tower of field extensions with E/L algebraic, and if M is an algebraically
  closed extension of K, then any embedding of L/K into M/K extends to an embedding of E/K.
  Known as the extension lemma in https://math.stackexchange.com/a/687914. -/
theorem surjective_restrictDomain_of_isAlgebraic {E : Type*}
    [Field E] [Algebra K E] [Algebra L E] [IsScalarTower K L E] [Algebra.IsAlgebraic L E] :
    Function.Surjective fun φ : E →ₐ[K] M ↦ φ.restrictDomain L :=
  fun f ↦ IntermediateField.exists_algHom_of_splits'
    (E := E) f fun s ↦ ⟨Algebra.IsIntegral.isIntegral s, IsAlgClosed.splits_codomain _⟩


@[deprecated (since := "2024-11-15")]
alias surjective_comp_algebraMap_of_isAlgebraic := surjective_restrictDomain_of_isAlgebraic


/-- Less general version of `lift`. -/
private noncomputable irreducible_def lift_aux : L →ₐ[K] M :=
  Classical.choice <| IntermediateField.nonempty_algHom_of_adjoin_splits
    (fun x _ ↦ ⟨Algebra.IsIntegral.isIntegral x, splits_codomain (minpoly K x)⟩)
    (IntermediateField.adjoin_univ K L)


private instance FractionRing.isAlgebraic :
    letI : IsDomain R := (NoZeroSMulDivisors.algebraMap_injective R S).isDomain _
    letI : Algebra (FractionRing R) (FractionRing S) := FractionRing.liftAlgebra R _
    Algebra.IsAlgebraic (FractionRing R) (FractionRing S) := by
  /-
    k : Type u
    inst✝¹⁵ : Field k
    K : Type u
    inst✝¹⁴ : Field K
    L : Type v
    M : Type w
    inst✝¹³ : Field L
    inst✝¹² : Algebra K L
    inst✝¹¹ : Field M
    inst✝¹⁰ : Algebra K M
    inst✝⁹ : IsAlgClosed M
    inst✝⁸ : Algebra.IsAlgebraic K L
    R : Type u
    inst✝⁷ : CommRing R
    S : Type v
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R M
    inst✝² : NoZeroSMulDivisors R S
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : Algebra.IsAlgebraic R S
    ⊢ Algebra.IsAlgebraic (FractionRing R) (FractionRing S)
  -/
  letI : IsDomain R := (NoZeroSMulDivisors.algebraMap_injective R S).isDomain _
  /-
    k : Type u
    inst✝¹⁵ : Field k
    K : Type u
    inst✝¹⁴ : Field K
    L : Type v
    M : Type w
    inst✝¹³ : Field L
    inst✝¹² : Algebra K L
    inst✝¹¹ : Field M
    inst✝¹⁰ : Algebra K M
    inst✝⁹ : IsAlgClosed M
    inst✝⁸ : Algebra.IsAlgebraic K L
    R : Type u
    inst✝⁷ : CommRing R
    S : Type v
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R M
    inst✝² : NoZeroSMulDivisors R S
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : Algebra.IsAlgebraic R S
    this : IsDomain R := Function.Injective.isDomain (algebraMap R S) (NoZeroSMulD …
    ⊢ Algebra.IsAlgebraic (FractionRing R) (FractionRing S)
  -/
  letI : Algebra (FractionRing R) (FractionRing S) := FractionRing.liftAlgebra R _
  /-
    k : Type u
    inst✝¹⁵ : Field k
    K : Type u
    inst✝¹⁴ : Field K
    L : Type v
    M : Type w
    inst✝¹³ : Field L
    inst✝¹² : Algebra K L
    inst✝¹¹ : Field M
    inst✝¹⁰ : Algebra K M
    inst✝⁹ : IsAlgClosed M
    inst✝⁸ : Algebra.IsAlgebraic K L
    R : Type u
    inst✝⁷ : CommRing R
    S : Type v
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R M
    inst✝² : NoZeroSMulDivisors R S
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : Algebra.IsAlgebraic R S
    this✝ : IsDomain R := Function.Injective.isDomain (algebraMap R S) (NoZeroSMul …
    this : Algebra (FractionRing R) (FractionRing S) := FractionRing.liftAlgebra R …
    ⊢ Algebra.IsAlgebraic (FractionRing R) (FractionRing S)
  -/
  have := FractionRing.isScalarTower_liftAlgebra R (FractionRing S)
  /-
    k : Type u
    inst✝¹⁵ : Field k
    K : Type u
    inst✝¹⁴ : Field K
    L : Type v
    M : Type w
    inst✝¹³ : Field L
    inst✝¹² : Algebra K L
    inst✝¹¹ : Field M
    inst✝¹⁰ : Algebra K M
    inst✝⁹ : IsAlgClosed M
    inst✝⁸ : Algebra.IsAlgebraic K L
    R : Type u
    inst✝⁷ : CommRing R
    S : Type v
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R M
    inst✝² : NoZeroSMulDivisors R S
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : Algebra.IsAlgebraic R S
    this✝¹ : IsDomain R := Function.Injective.isDomain (algebraMap R S) (NoZeroSMu …
    this✝ : Algebra (FractionRing R) (FractionRing S) := FractionRing.liftAlgebra  …
    this : IsScalarTower R (FractionRing R) (FractionRing S)
    ⊢ Algebra.IsAlgebraic (FractionRing R) (FractionRing S)
  -/
  have := (IsFractionRing.isAlgebraic_iff' R S (FractionRing S)).1 inferInstance
  /-
    k : Type u
    inst✝¹⁵ : Field k
    K : Type u
    inst✝¹⁴ : Field K
    L : Type v
    M : Type w
    inst✝¹³ : Field L
    inst✝¹² : Algebra K L
    inst✝¹¹ : Field M
    inst✝¹⁰ : Algebra K M
    inst✝⁹ : IsAlgClosed M
    inst✝⁸ : Algebra.IsAlgebraic K L
    R : Type u
    inst✝⁷ : CommRing R
    S : Type v
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R M
    inst✝² : NoZeroSMulDivisors R S
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : Algebra.IsAlgebraic R S
    this✝² : IsDomain R := Function.Injective.isDomain (algebraMap R S) (NoZeroSMu …
    this✝¹ : Algebra (FractionRing R) (FractionRing S) := FractionRing.liftAlgebra …
    this✝ : IsScalarTower R (FractionRing R) (FractionRing S)
    this : Algebra.IsAlgebraic R (FractionRing S)
    ⊢ Algebra.IsAlgebraic (FractionRing R) (FractionRing S)
  -/
  constructor
  /-
    case isAlgebraic
    k : Type u
    inst✝¹⁵ : Field k
    K : Type u
    inst✝¹⁴ : Field K
    L : Type v
    M : Type w
    inst✝¹³ : Field L
    inst✝¹² : Algebra K L
    inst✝¹¹ : Field M
    inst✝¹⁰ : Algebra K M
    inst✝⁹ : IsAlgClosed M
    inst✝⁸ : Algebra.IsAlgebraic K L
    R : Type u
    inst✝⁷ : CommRing R
    S : Type v
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R M
    inst✝² : NoZeroSMulDivisors R S
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : Algebra.IsAlgebraic R S
    this✝² : IsDomain R := Function.Injective.isDomain (algebraMap R S) (NoZeroSMu …
    this✝¹ : Algebra (FractionRing R) (FractionRing S) := FractionRing.liftAlgebra …
    this✝ : IsScalarTower R (FractionRing R) (FractionRing S)
    this : Algebra.IsAlgebraic R (FractionRing S)
    ⊢ ∀ (x : FractionRing S), IsAlgebraic (FractionRing R) x
  -/
  intro
  exact (IsFractionRing.isAlgebraic_iff R (FractionRing R) (FractionRing S)).1
      (Algebra.IsAlgebraic.isAlgebraic _)


/-- A (random) homomorphism from an algebraic extension of R into an algebraically
  closed extension of R. -/
@[stacks 09GU]
noncomputable irreducible_def lift : S →ₐ[R] M := by
  /-
    k : Type u
    inst✝¹⁵ : Field k
    K : Type u
    inst✝¹⁴ : Field K
    L : Type v
    M : Type w
    inst✝¹³ : Field L
    inst✝¹² : Algebra K L
    inst✝¹¹ : Field M
    inst✝¹⁰ : Algebra K M
    inst✝⁹ : IsAlgClosed M
    inst✝⁸ : Algebra.IsAlgebraic K L
    R : Type u
    inst✝⁷ : CommRing R
    S : Type v
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R M
    inst✝² : NoZeroSMulDivisors R S
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : Algebra.IsAlgebraic R S
    ⊢ AlgHom R S M
  -/
  letI : IsDomain R := (NoZeroSMulDivisors.algebraMap_injective R S).isDomain _
  /-
    k : Type u
    inst✝¹⁵ : Field k
    K : Type u
    inst✝¹⁴ : Field K
    L : Type v
    M : Type w
    inst✝¹³ : Field L
    inst✝¹² : Algebra K L
    inst✝¹¹ : Field M
    inst✝¹⁰ : Algebra K M
    inst✝⁹ : IsAlgClosed M
    inst✝⁸ : Algebra.IsAlgebraic K L
    R : Type u
    inst✝⁷ : CommRing R
    S : Type v
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R M
    inst✝² : NoZeroSMulDivisors R S
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : Algebra.IsAlgebraic R S
    this : IsDomain R := Function.Injective.isDomain (algebraMap R S) (NoZeroSMulD …
    ⊢ AlgHom R S M
  -/
  letI := FractionRing.liftAlgebra R M
  /-
    k : Type u
    inst✝¹⁵ : Field k
    K : Type u
    inst✝¹⁴ : Field K
    L : Type v
    M : Type w
    inst✝¹³ : Field L
    inst✝¹² : Algebra K L
    inst✝¹¹ : Field M
    inst✝¹⁰ : Algebra K M
    inst✝⁹ : IsAlgClosed M
    inst✝⁸ : Algebra.IsAlgebraic K L
    R : Type u
    inst✝⁷ : CommRing R
    S : Type v
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R M
    inst✝² : NoZeroSMulDivisors R S
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : Algebra.IsAlgebraic R S
    this✝ : IsDomain R := Function.Injective.isDomain (algebraMap R S) (NoZeroSMul …
    this : Algebra (FractionRing R) M := FractionRing.liftAlgebra R M
    ⊢ AlgHom R S M
  -/
  letI := FractionRing.liftAlgebra R (FractionRing S)
  /-
    k : Type u
    inst✝¹⁵ : Field k
    K : Type u
    inst✝¹⁴ : Field K
    L : Type v
    M : Type w
    inst✝¹³ : Field L
    inst✝¹² : Algebra K L
    inst✝¹¹ : Field M
    inst✝¹⁰ : Algebra K M
    inst✝⁹ : IsAlgClosed M
    inst✝⁸ : Algebra.IsAlgebraic K L
    R : Type u
    inst✝⁷ : CommRing R
    S : Type v
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R M
    inst✝² : NoZeroSMulDivisors R S
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : Algebra.IsAlgebraic R S
    this✝¹ : IsDomain R := Function.Injective.isDomain (algebraMap R S) (NoZeroSMu …
    this✝ : Algebra (FractionRing R) M := FractionRing.liftAlgebra R M
    this : Algebra (FractionRing R) (FractionRing S) := FractionRing.liftAlgebra R …
    ⊢ AlgHom R S M
  -/
  have := FractionRing.isScalarTower_liftAlgebra R M
  /-
    k : Type u
    inst✝¹⁵ : Field k
    K : Type u
    inst✝¹⁴ : Field K
    L : Type v
    M : Type w
    inst✝¹³ : Field L
    inst✝¹² : Algebra K L
    inst✝¹¹ : Field M
    inst✝¹⁰ : Algebra K M
    inst✝⁹ : IsAlgClosed M
    inst✝⁸ : Algebra.IsAlgebraic K L
    R : Type u
    inst✝⁷ : CommRing R
    S : Type v
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R M
    inst✝² : NoZeroSMulDivisors R S
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : Algebra.IsAlgebraic R S
    this✝² : IsDomain R := Function.Injective.isDomain (algebraMap R S) (NoZeroSMu …
    this✝¹ : Algebra (FractionRing R) M := FractionRing.liftAlgebra R M
    this✝ : Algebra (FractionRing R) (FractionRing S) := FractionRing.liftAlgebra  …
    this : IsScalarTower R (FractionRing R) M
    ⊢ AlgHom R S M
  -/
  have := FractionRing.isScalarTower_liftAlgebra R (FractionRing S)
  /-
    k : Type u
    inst✝¹⁵ : Field k
    K : Type u
    inst✝¹⁴ : Field K
    L : Type v
    M : Type w
    inst✝¹³ : Field L
    inst✝¹² : Algebra K L
    inst✝¹¹ : Field M
    inst✝¹⁰ : Algebra K M
    inst✝⁹ : IsAlgClosed M
    inst✝⁸ : Algebra.IsAlgebraic K L
    R : Type u
    inst✝⁷ : CommRing R
    S : Type v
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R M
    inst✝² : NoZeroSMulDivisors R S
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : Algebra.IsAlgebraic R S
    this✝³ : IsDomain R := Function.Injective.isDomain (algebraMap R S) (NoZeroSMu …
    this✝² : Algebra (FractionRing R) M := FractionRing.liftAlgebra R M
    this✝¹ : Algebra (FractionRing R) (FractionRing S) := FractionRing.liftAlgebra …
    this✝ : IsScalarTower R (FractionRing R) M
    this : IsScalarTower R (FractionRing R) (FractionRing S)
    ⊢ AlgHom R S M
  -/
  let f : FractionRing S →ₐ[FractionRing R] M := lift_aux (FractionRing R) (FractionRing S) M
  /-
    k : Type u
    inst✝¹⁵ : Field k
    K : Type u
    inst✝¹⁴ : Field K
    L : Type v
    M : Type w
    inst✝¹³ : Field L
    inst✝¹² : Algebra K L
    inst✝¹¹ : Field M
    inst✝¹⁰ : Algebra K M
    inst✝⁹ : IsAlgClosed M
    inst✝⁸ : Algebra.IsAlgebraic K L
    R : Type u
    inst✝⁷ : CommRing R
    S : Type v
    inst✝⁶ : CommRing S
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R M
    inst✝² : NoZeroSMulDivisors R S
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : Algebra.IsAlgebraic R S
    this✝³ : IsDomain R := Function.Injective.isDomain (algebraMap R S) (NoZeroSMu …
    this✝² : Algebra (FractionRing R) M := FractionRing.liftAlgebra R M
    this✝¹ : Algebra (FractionRing R) (FractionRing S) := FractionRing.liftAlgebra …
    this✝ : IsScalarTower R (FractionRing R) M
    this : IsScalarTower R (FractionRing R) (FractionRing S)
    f : AlgHom (FractionRing R) (FractionRing S) M := IsAlgClosed.lift_aux (Fracti …
    ⊢ AlgHom R S M
  -/
  exact (f.restrictScalars R).comp ((Algebra.ofId S (FractionRing S)).restrictScalars R)
  /-
    🎉 no goals
  -/


noncomputable instance (priority := 100) perfectRing (p : ℕ) [Fact p.Prime] [CharP k p]
    [IsAlgClosed k] : PerfectRing k p :=
  PerfectRing.ofSurjective k p fun _ => IsAlgClosed.exists_pow_nat_eq _ <| NeZero.pos p


noncomputable instance (priority := 100) perfectField [IsAlgClosed k] : PerfectField k := by
  /-
    k : Type u
    inst✝¹⁶ : Field k
    K : Type u
    inst✝¹⁵ : Field K
    L : Type v
    M : Type w
    inst✝¹⁴ : Field L
    inst✝¹³ : Algebra K L
    inst✝¹² : Field M
    inst✝¹¹ : Algebra K M
    inst✝¹⁰ : IsAlgClosed M
    inst✝⁹ : Algebra.IsAlgebraic K L
    R : Type u
    inst✝⁸ : CommRing R
    S : Type v
    inst✝⁷ : CommRing S
    inst✝⁶ : IsDomain S
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R M
    inst✝³ : NoZeroSMulDivisors R S
    inst✝² : NoZeroSMulDivisors R M
    inst✝¹ : Algebra.IsAlgebraic R S
    inst✝ : IsAlgClosed k
    ⊢ PerfectField k
  -/
  obtain _ | ⟨p, _, _⟩ := CharP.exists' k
  /-
    case inl
    k : Type u
    inst✝¹⁶ : Field k
    K : Type u
    inst✝¹⁵ : Field K
    L : Type v
    M : Type w
    inst✝¹⁴ : Field L
    inst✝¹³ : Algebra K L
    inst✝¹² : Field M
    inst✝¹¹ : Algebra K M
    inst✝¹⁰ : IsAlgClosed M
    inst✝⁹ : Algebra.IsAlgebraic K L
    R : Type u
    inst✝⁸ : CommRing R
    S : Type v
    inst✝⁷ : CommRing S
    inst✝⁶ : IsDomain S
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R M
    inst✝³ : NoZeroSMulDivisors R S
    inst✝² : NoZeroSMulDivisors R M
    inst✝¹ : Algebra.IsAlgebraic R S
    inst✝ : IsAlgClosed k
    h✝ : CharZero k
    ⊢ PerfectField k
  -/
  exacts [.ofCharZero, PerfectRing.toPerfectField k p]
  /-
    🎉 no goals
  -/


/-- Algebraically closed fields are infinite since `Xⁿ⁺¹ - 1` is separable when `#K = n` -/
instance (priority := 500) {K : Type*} [Field K] [IsAlgClosed K] : Infinite K := by
  /-
    k : Type u
    inst✝¹⁷ : Field k
    K✝ : Type u
    inst✝¹⁶ : Field K✝
    L : Type v
    M : Type w
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra K✝ L
    inst✝¹³ : Field M
    inst✝¹² : Algebra K✝ M
    inst✝¹¹ : IsAlgClosed M
    inst✝¹⁰ : Algebra.IsAlgebraic K✝ L
    R : Type u
    inst✝⁹ : CommRing R
    S : Type v
    inst✝⁸ : CommRing S
    inst✝⁷ : IsDomain S
    inst✝⁶ : Algebra R S
    inst✝⁵ : Algebra R M
    inst✝⁴ : NoZeroSMulDivisors R S
    inst✝³ : NoZeroSMulDivisors R M
    inst✝² : Algebra.IsAlgebraic R S
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : IsAlgClosed K
    ⊢ Infinite K
  -/
  apply Infinite.of_not_fintype
  /-
    case h
    k : Type u
    inst✝¹⁷ : Field k
    K✝ : Type u
    inst✝¹⁶ : Field K✝
    L : Type v
    M : Type w
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra K✝ L
    inst✝¹³ : Field M
    inst✝¹² : Algebra K✝ M
    inst✝¹¹ : IsAlgClosed M
    inst✝¹⁰ : Algebra.IsAlgebraic K✝ L
    R : Type u
    inst✝⁹ : CommRing R
    S : Type v
    inst✝⁸ : CommRing S
    inst✝⁷ : IsDomain S
    inst✝⁶ : Algebra R S
    inst✝⁵ : Algebra R M
    inst✝⁴ : NoZeroSMulDivisors R S
    inst✝³ : NoZeroSMulDivisors R M
    inst✝² : Algebra.IsAlgebraic R S
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : IsAlgClosed K
    ⊢ Fintype K → False
  -/
  intro hfin
  /-
    case h
    k : Type u
    inst✝¹⁷ : Field k
    K✝ : Type u
    inst✝¹⁶ : Field K✝
    L : Type v
    M : Type w
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra K✝ L
    inst✝¹³ : Field M
    inst✝¹² : Algebra K✝ M
    inst✝¹¹ : IsAlgClosed M
    inst✝¹⁰ : Algebra.IsAlgebraic K✝ L
    R : Type u
    inst✝⁹ : CommRing R
    S : Type v
    inst✝⁸ : CommRing S
    inst✝⁷ : IsDomain S
    inst✝⁶ : Algebra R S
    inst✝⁵ : Algebra R M
    inst✝⁴ : NoZeroSMulDivisors R S
    inst✝³ : NoZeroSMulDivisors R M
    inst✝² : Algebra.IsAlgebraic R S
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : IsAlgClosed K
    hfin : Fintype K
    ⊢ False
  -/
  set n := Fintype.card K
  /-
    case h
    k : Type u
    inst✝¹⁷ : Field k
    K✝ : Type u
    inst✝¹⁶ : Field K✝
    L : Type v
    M : Type w
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra K✝ L
    inst✝¹³ : Field M
    inst✝¹² : Algebra K✝ M
    inst✝¹¹ : IsAlgClosed M
    inst✝¹⁰ : Algebra.IsAlgebraic K✝ L
    R : Type u
    inst✝⁹ : CommRing R
    S : Type v
    inst✝⁸ : CommRing S
    inst✝⁷ : IsDomain S
    inst✝⁶ : Algebra R S
    inst✝⁵ : Algebra R M
    inst✝⁴ : NoZeroSMulDivisors R S
    inst✝³ : NoZeroSMulDivisors R M
    inst✝² : Algebra.IsAlgebraic R S
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : IsAlgClosed K
    hfin : Fintype K
    n : Nat := Fintype.card K
    ⊢ False
  -/
  set f := (X : K[X]) ^ (n + 1) - 1
  /-
    case h
    k : Type u
    inst✝¹⁷ : Field k
    K✝ : Type u
    inst✝¹⁶ : Field K✝
    L : Type v
    M : Type w
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra K✝ L
    inst✝¹³ : Field M
    inst✝¹² : Algebra K✝ M
    inst✝¹¹ : IsAlgClosed M
    inst✝¹⁰ : Algebra.IsAlgebraic K✝ L
    R : Type u
    inst✝⁹ : CommRing R
    S : Type v
    inst✝⁸ : CommRing S
    inst✝⁷ : IsDomain S
    inst✝⁶ : Algebra R S
    inst✝⁵ : Algebra R M
    inst✝⁴ : NoZeroSMulDivisors R S
    inst✝³ : NoZeroSMulDivisors R M
    inst✝² : Algebra.IsAlgebraic R S
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : IsAlgClosed K
    hfin : Fintype K
    n : Nat := Fintype.card K
    f : Polynomial K := HSub.hSub (HPow.hPow Polynomial.X (HAdd.hAdd n 1)) 1
    ⊢ False
  -/
  have hfsep : Separable f := separable_X_pow_sub_C 1 (by simp [n]) one_ne_zero
  /-
    case h
    k : Type u
    inst✝¹⁷ : Field k
    K✝ : Type u
    inst✝¹⁶ : Field K✝
    L : Type v
    M : Type w
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra K✝ L
    inst✝¹³ : Field M
    inst✝¹² : Algebra K✝ M
    inst✝¹¹ : IsAlgClosed M
    inst✝¹⁰ : Algebra.IsAlgebraic K✝ L
    R : Type u
    inst✝⁹ : CommRing R
    S : Type v
    inst✝⁸ : CommRing S
    inst✝⁷ : IsDomain S
    inst✝⁶ : Algebra R S
    inst✝⁵ : Algebra R M
    inst✝⁴ : NoZeroSMulDivisors R S
    inst✝³ : NoZeroSMulDivisors R M
    inst✝² : Algebra.IsAlgebraic R S
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : IsAlgClosed K
    hfin : Fintype K
    n : Nat := Fintype.card K
    f : Polynomial K := HSub.hSub (HPow.hPow Polynomial.X (HAdd.hAdd n 1)) 1
    hfsep : f.Separable
    ⊢ False
  -/
  apply Nat.not_succ_le_self (Fintype.card K)
  have hroot : n.succ = Fintype.card (f.rootSet K) := by
    erw [card_rootSet_eq_natDegree hfsep (IsAlgClosed.splits_domain _), natDegree_X_pow_sub_C]
  /-
    case h
    k : Type u
    inst✝¹⁷ : Field k
    K✝ : Type u
    inst✝¹⁶ : Field K✝
    L : Type v
    M : Type w
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra K✝ L
    inst✝¹³ : Field M
    inst✝¹² : Algebra K✝ M
    inst✝¹¹ : IsAlgClosed M
    inst✝¹⁰ : Algebra.IsAlgebraic K✝ L
    R : Type u
    inst✝⁹ : CommRing R
    S : Type v
    inst✝⁸ : CommRing S
    inst✝⁷ : IsDomain S
    inst✝⁶ : Algebra R S
    inst✝⁵ : Algebra R M
    inst✝⁴ : NoZeroSMulDivisors R S
    inst✝³ : NoZeroSMulDivisors R M
    inst✝² : Algebra.IsAlgebraic R S
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : IsAlgClosed K
    hfin : Fintype K
    n : Nat := Fintype.card K
    f : Polynomial K := HSub.hSub (HPow.hPow Polynomial.X (HAdd.hAdd n 1)) 1
    hfsep : f.Separable
    hroot : Eq n.succ (Fintype.card ↑(f.rootSet K))
    ⊢ LE.le (Fintype.card K).succ (Fintype.card K)
  -/
  rw [hroot]
  /-
    case h
    k : Type u
    inst✝¹⁷ : Field k
    K✝ : Type u
    inst✝¹⁶ : Field K✝
    L : Type v
    M : Type w
    inst✝¹⁵ : Field L
    inst✝¹⁴ : Algebra K✝ L
    inst✝¹³ : Field M
    inst✝¹² : Algebra K✝ M
    inst✝¹¹ : IsAlgClosed M
    inst✝¹⁰ : Algebra.IsAlgebraic K✝ L
    R : Type u
    inst✝⁹ : CommRing R
    S : Type v
    inst✝⁸ : CommRing S
    inst✝⁷ : IsDomain S
    inst✝⁶ : Algebra R S
    inst✝⁵ : Algebra R M
    inst✝⁴ : NoZeroSMulDivisors R S
    inst✝³ : NoZeroSMulDivisors R M
    inst✝² : Algebra.IsAlgebraic R S
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : IsAlgClosed K
    hfin : Fintype K
    n : Nat := Fintype.card K
    f : Polynomial K := HSub.hSub (HPow.hPow Polynomial.X (HAdd.hAdd n 1)) 1
    hfsep : f.Separable
    hroot : Eq n.succ (Fintype.card ↑(f.rootSet K))
    ⊢ LE.le (Fintype.card ↑(f.rootSet K)) (Fintype.card K)
  -/
  exact Fintype.card_le_of_injective _ Subtype.coe_injective
  /-
    🎉 no goals
  -/


/-- A (random) isomorphism between two algebraic closures of `R`. -/
@[stacks 09GV]
noncomputable def equiv : L ≃ₐ[R] M :=
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): added to replace local instance above
  haveI : IsAlgClosed L := IsAlgClosure.isAlgClosed R
  haveI : IsAlgClosed M := IsAlgClosure.isAlgClosed R
  AlgEquiv.ofBijective _ (IsAlgClosure.isAlgebraic.algHom_bijective₂
    (IsAlgClosed.lift : L →ₐ[R] M)
    (IsAlgClosed.lift : M →ₐ[R] L)).1


/-- If `J` is an algebraic extension of `K` and `L` is an algebraic closure of `J`, then it is
  also an algebraic closure of `K`. -/
theorem ofAlgebraic [hKJ : Algebra.IsAlgebraic K J] : IsAlgClosure K L :=
  ⟨IsAlgClosure.isAlgClosed J, hKJ.trans⟩


/-- A (random) isomorphism between an algebraic closure of `R` and an algebraic closure of
  an algebraic extension of `R` -/
noncomputable def equivOfAlgebraic' [Nontrivial S] [NoZeroSMulDivisors R S]
    [Algebra.IsAlgebraic R L] : L ≃ₐ[R] M := by
  letI : NoZeroSMulDivisors R L := NoZeroSMulDivisors.of_algebraMap_injective <| by
    rw [IsScalarTower.algebraMap_eq R S L]
    exact (Function.Injective.comp (NoZeroSMulDivisors.algebraMap_injective S L)
            (NoZeroSMulDivisors.algebraMap_injective R S) : _)
  letI : IsAlgClosure R L :=
    { isAlgClosed := IsAlgClosure.isAlgClosed S
      isAlgebraic := ‹_› }
  /-
    k : Type u
    inst✝²⁵ : Field k
    K : Type u_1
    J : Type u_2
    R : Type u
    S : Type u_3
    L : Type v
    M : Type w
    inst✝²⁴ : Field K
    inst✝²³ : Field J
    inst✝²² : CommRing R
    inst✝²¹ : CommRing S
    inst✝²⁰ : Field L
    inst✝¹⁹ : Field M
    inst✝¹⁸ : Algebra R M
    inst✝¹⁷ : NoZeroSMulDivisors R M
    inst✝¹⁶ : IsAlgClosure R M
    inst✝¹⁵ : Algebra K M
    inst✝¹⁴ : IsAlgClosure K M
    inst✝¹³ : Algebra S L
    inst✝¹² : NoZeroSMulDivisors S L
    inst✝¹¹ : IsAlgClosure S L
    inst✝¹⁰ : Algebra R S
    inst✝⁹ : Algebra R L
    inst✝⁸ : IsScalarTower R S L
    inst✝⁷ : Algebra K J
    inst✝⁶ : Algebra J L
    inst✝⁵ : IsAlgClosure J L
    inst✝⁴ : Algebra K L
    inst✝³ : IsScalarTower K J L
    inst✝² : Nontrivial S
    inst✝¹ : NoZeroSMulDivisors R S
    inst✝ : Algebra.IsAlgebraic R L
    this✝ : NoZeroSMulDivisors R L := NoZeroSMulDivisors.of_algebraMap_injective ( …
    this : IsAlgClosure R L := { isAlgClosed := IsAlgClosure.isAlgClosed S, isAlge …
    ⊢ AlgEquiv R L M
  -/
  exact IsAlgClosure.equiv _ _ _
  /-
    🎉 no goals
  -/


/-- A (random) isomorphism between an algebraic closure of `K` and an algebraic closure
  of an algebraic extension of `K` -/
noncomputable def equivOfAlgebraic [hKJ : Algebra.IsAlgebraic K J] : L ≃ₐ[K] M :=
  have : Algebra.IsAlgebraic K L := hKJ.trans
  equivOfAlgebraic' K J _ _


/-- Used in the definition of `equivOfEquiv` -/
noncomputable def equivOfEquivAux (hSR : S ≃+* R) :
    { e : L ≃+* M // e.toRingHom.comp (algebraMap S L) = (algebraMap R M).comp hSR.toRingHom } := by
  /-
    k : Type u
    inst✝¹⁴ : Field k
    K : Type u_1
    J : Type u_2
    R : Type u
    S : Type u_3
    L : Type v
    M : Type w
    inst✝¹³ : Field K
    inst✝¹² : Field J
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Field L
    inst✝⁸ : Field M
    inst✝⁷ : Algebra R M
    inst✝⁶ : NoZeroSMulDivisors R M
    inst✝⁵ : IsAlgClosure R M
    inst✝⁴ : Algebra K M
    inst✝³ : IsAlgClosure K M
    inst✝² : Algebra S L
    inst✝¹ : NoZeroSMulDivisors S L
    inst✝ : IsAlgClosure S L
    hSR : RingEquiv S R
    ⊢ Subtype fun e => Eq (e.toRingHom.comp (algebraMap S L)) ((algebraMap R M).co …
  -/
  letI : Algebra R S := RingHom.toAlgebra hSR.symm.toRingHom
  /-
    k : Type u
    inst✝¹⁴ : Field k
    K : Type u_1
    J : Type u_2
    R : Type u
    S : Type u_3
    L : Type v
    M : Type w
    inst✝¹³ : Field K
    inst✝¹² : Field J
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Field L
    inst✝⁸ : Field M
    inst✝⁷ : Algebra R M
    inst✝⁶ : NoZeroSMulDivisors R M
    inst✝⁵ : IsAlgClosure R M
    inst✝⁴ : Algebra K M
    inst✝³ : IsAlgClosure K M
    inst✝² : Algebra S L
    inst✝¹ : NoZeroSMulDivisors S L
    inst✝ : IsAlgClosure S L
    hSR : RingEquiv S R
    this : Algebra R S := hSR.symm.toRingHom.toAlgebra
    ⊢ Subtype fun e => Eq (e.toRingHom.comp (algebraMap S L)) ((algebraMap R M).co …
  -/
  letI : Algebra S R := RingHom.toAlgebra hSR.toRingHom
  /-
    k : Type u
    inst✝¹⁴ : Field k
    K : Type u_1
    J : Type u_2
    R : Type u
    S : Type u_3
    L : Type v
    M : Type w
    inst✝¹³ : Field K
    inst✝¹² : Field J
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Field L
    inst✝⁸ : Field M
    inst✝⁷ : Algebra R M
    inst✝⁶ : NoZeroSMulDivisors R M
    inst✝⁵ : IsAlgClosure R M
    inst✝⁴ : Algebra K M
    inst✝³ : IsAlgClosure K M
    inst✝² : Algebra S L
    inst✝¹ : NoZeroSMulDivisors S L
    inst✝ : IsAlgClosure S L
    hSR : RingEquiv S R
    this✝ : Algebra R S := hSR.symm.toRingHom.toAlgebra
    this : Algebra S R := hSR.toRingHom.toAlgebra
    ⊢ Subtype fun e => Eq (e.toRingHom.comp (algebraMap S L)) ((algebraMap R M).co …
  -/
  letI : IsDomain R := (NoZeroSMulDivisors.algebraMap_injective R M).isDomain _
  /-
    k : Type u
    inst✝¹⁴ : Field k
    K : Type u_1
    J : Type u_2
    R : Type u
    S : Type u_3
    L : Type v
    M : Type w
    inst✝¹³ : Field K
    inst✝¹² : Field J
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Field L
    inst✝⁸ : Field M
    inst✝⁷ : Algebra R M
    inst✝⁶ : NoZeroSMulDivisors R M
    inst✝⁵ : IsAlgClosure R M
    inst✝⁴ : Algebra K M
    inst✝³ : IsAlgClosure K M
    inst✝² : Algebra S L
    inst✝¹ : NoZeroSMulDivisors S L
    inst✝ : IsAlgClosure S L
    hSR : RingEquiv S R
    this✝¹ : Algebra R S := hSR.symm.toRingHom.toAlgebra
    this✝ : Algebra S R := hSR.toRingHom.toAlgebra
    this : IsDomain R := Function.Injective.isDomain (algebraMap R M) (NoZeroSMulD …
    ⊢ Subtype fun e => Eq (e.toRingHom.comp (algebraMap S L)) ((algebraMap R M).co …
  -/
  letI : IsDomain S := (NoZeroSMulDivisors.algebraMap_injective S L).isDomain _
  /-
    k : Type u
    inst✝¹⁴ : Field k
    K : Type u_1
    J : Type u_2
    R : Type u
    S : Type u_3
    L : Type v
    M : Type w
    inst✝¹³ : Field K
    inst✝¹² : Field J
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Field L
    inst✝⁸ : Field M
    inst✝⁷ : Algebra R M
    inst✝⁶ : NoZeroSMulDivisors R M
    inst✝⁵ : IsAlgClosure R M
    inst✝⁴ : Algebra K M
    inst✝³ : IsAlgClosure K M
    inst✝² : Algebra S L
    inst✝¹ : NoZeroSMulDivisors S L
    inst✝ : IsAlgClosure S L
    hSR : RingEquiv S R
    this✝² : Algebra R S := hSR.symm.toRingHom.toAlgebra
    this✝¹ : Algebra S R := hSR.toRingHom.toAlgebra
    this✝ : IsDomain R := Function.Injective.isDomain (algebraMap R M) (NoZeroSMul …
    this : IsDomain S := Function.Injective.isDomain (algebraMap S L) (NoZeroSMulD …
    ⊢ Subtype fun e => Eq (e.toRingHom.comp (algebraMap S L)) ((algebraMap R M).co …
  -/
  letI : Algebra R L := RingHom.toAlgebra ((algebraMap S L).comp (algebraMap R S))
  /-
    k : Type u
    inst✝¹⁴ : Field k
    K : Type u_1
    J : Type u_2
    R : Type u
    S : Type u_3
    L : Type v
    M : Type w
    inst✝¹³ : Field K
    inst✝¹² : Field J
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Field L
    inst✝⁸ : Field M
    inst✝⁷ : Algebra R M
    inst✝⁶ : NoZeroSMulDivisors R M
    inst✝⁵ : IsAlgClosure R M
    inst✝⁴ : Algebra K M
    inst✝³ : IsAlgClosure K M
    inst✝² : Algebra S L
    inst✝¹ : NoZeroSMulDivisors S L
    inst✝ : IsAlgClosure S L
    hSR : RingEquiv S R
    this✝³ : Algebra R S := hSR.symm.toRingHom.toAlgebra
    this✝² : Algebra S R := hSR.toRingHom.toAlgebra
    this✝¹ : IsDomain R := Function.Injective.isDomain (algebraMap R M) (NoZeroSMu …
    this✝ : IsDomain S := Function.Injective.isDomain (algebraMap S L) (NoZeroSMul …
    this : Algebra R L := ((algebraMap S L).comp (algebraMap R S)).toAlgebra
    ⊢ Subtype fun e => Eq (e.toRingHom.comp (algebraMap S L)) ((algebraMap R M).co …
  -/
  haveI : IsScalarTower R S L := IsScalarTower.of_algebraMap_eq fun _ => rfl
  haveI : IsScalarTower S R L :=
    IsScalarTower.of_algebraMap_eq (by simp [RingHom.algebraMap_toAlgebra])
  /-
    k : Type u
    inst✝¹⁴ : Field k
    K : Type u_1
    J : Type u_2
    R : Type u
    S : Type u_3
    L : Type v
    M : Type w
    inst✝¹³ : Field K
    inst✝¹² : Field J
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Field L
    inst✝⁸ : Field M
    inst✝⁷ : Algebra R M
    inst✝⁶ : NoZeroSMulDivisors R M
    inst✝⁵ : IsAlgClosure R M
    inst✝⁴ : Algebra K M
    inst✝³ : IsAlgClosure K M
    inst✝² : Algebra S L
    inst✝¹ : NoZeroSMulDivisors S L
    inst✝ : IsAlgClosure S L
    hSR : RingEquiv S R
    this✝⁵ : Algebra R S := hSR.symm.toRingHom.toAlgebra
    this✝⁴ : Algebra S R := hSR.toRingHom.toAlgebra
    this✝³ : IsDomain R := Function.Injective.isDomain (algebraMap R M) (NoZeroSMu …
    this✝² : IsDomain S := Function.Injective.isDomain (algebraMap S L) (NoZeroSMu …
    this✝¹ : Algebra R L := ((algebraMap S L).comp (algebraMap R S)).toAlgebra
    this✝ : IsScalarTower R S L
    this : IsScalarTower S R L
    ⊢ Subtype fun e => Eq (e.toRingHom.comp (algebraMap S L)) ((algebraMap R M).co …
  -/
  haveI : NoZeroSMulDivisors R S := NoZeroSMulDivisors.of_algebraMap_injective hSR.symm.injective
  have : Algebra.IsAlgebraic R L := (IsAlgClosure.isAlgebraic.extendScalars
    (show Function.Injective (algebraMap S R) from hSR.injective))
  refine
    ⟨equivOfAlgebraic' R S L M, ?_⟩
  /-
    k : Type u
    inst✝¹⁴ : Field k
    K : Type u_1
    J : Type u_2
    R : Type u
    S : Type u_3
    L : Type v
    M : Type w
    inst✝¹³ : Field K
    inst✝¹² : Field J
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Field L
    inst✝⁸ : Field M
    inst✝⁷ : Algebra R M
    inst✝⁶ : NoZeroSMulDivisors R M
    inst✝⁵ : IsAlgClosure R M
    inst✝⁴ : Algebra K M
    inst✝³ : IsAlgClosure K M
    inst✝² : Algebra S L
    inst✝¹ : NoZeroSMulDivisors S L
    inst✝ : IsAlgClosure S L
    hSR : RingEquiv S R
    this✝⁷ : Algebra R S := hSR.symm.toRingHom.toAlgebra
    this✝⁶ : Algebra S R := hSR.toRingHom.toAlgebra
    this✝⁵ : IsDomain R := Function.Injective.isDomain (algebraMap R M) (NoZeroSMu …
    this✝⁴ : IsDomain S := Function.Injective.isDomain (algebraMap S L) (NoZeroSMu …
    this✝³ : Algebra R L := ((algebraMap S L).comp (algebraMap R S)).toAlgebra
    this✝² : IsScalarTower R S L
    this✝¹ : IsScalarTower S R L
    this✝ : NoZeroSMulDivisors R S
    this : Algebra.IsAlgebraic R L
    ⊢ Eq ((↑(IsAlgClosure.equivOfAlgebraic' R S L M)).toRingHom.comp (algebraMap S …
  -/
  ext x
  simp only [RingEquiv.toRingHom_eq_coe, Function.comp_apply, RingHom.coe_comp,
    AlgEquiv.coe_ringEquiv, RingEquiv.coe_toRingHom]
  /-
    case a
    k : Type u
    inst✝¹⁴ : Field k
    K : Type u_1
    J : Type u_2
    R : Type u
    S : Type u_3
    L : Type v
    M : Type w
    inst✝¹³ : Field K
    inst✝¹² : Field J
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Field L
    inst✝⁸ : Field M
    inst✝⁷ : Algebra R M
    inst✝⁶ : NoZeroSMulDivisors R M
    inst✝⁵ : IsAlgClosure R M
    inst✝⁴ : Algebra K M
    inst✝³ : IsAlgClosure K M
    inst✝² : Algebra S L
    inst✝¹ : NoZeroSMulDivisors S L
    inst✝ : IsAlgClosure S L
    hSR : RingEquiv S R
    this✝⁷ : Algebra R S := hSR.symm.toRingHom.toAlgebra
    this✝⁶ : Algebra S R := hSR.toRingHom.toAlgebra
    this✝⁵ : IsDomain R := Function.Injective.isDomain (algebraMap R M) (NoZeroSMu …
    this✝⁴ : IsDomain S := Function.Injective.isDomain (algebraMap S L) (NoZeroSMu …
    this✝³ : Algebra R L := ((algebraMap S L).comp (algebraMap R S)).toAlgebra
    this✝² : IsScalarTower R S L
    this✝¹ : IsScalarTower S R L
    this✝ : NoZeroSMulDivisors R S
    this : Algebra.IsAlgebraic R L
    x : S
    ⊢ Eq ((IsAlgClosure.equivOfAlgebraic' R S L M) ((algebraMap S L) x)) ((algebra …
  -/
  conv_lhs => rw [← hSR.symm_apply_apply x]
  /-
    case a
    k : Type u
    inst✝¹⁴ : Field k
    K : Type u_1
    J : Type u_2
    R : Type u
    S : Type u_3
    L : Type v
    M : Type w
    inst✝¹³ : Field K
    inst✝¹² : Field J
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Field L
    inst✝⁸ : Field M
    inst✝⁷ : Algebra R M
    inst✝⁶ : NoZeroSMulDivisors R M
    inst✝⁵ : IsAlgClosure R M
    inst✝⁴ : Algebra K M
    inst✝³ : IsAlgClosure K M
    inst✝² : Algebra S L
    inst✝¹ : NoZeroSMulDivisors S L
    inst✝ : IsAlgClosure S L
    hSR : RingEquiv S R
    this✝⁷ : Algebra R S := hSR.symm.toRingHom.toAlgebra
    this✝⁶ : Algebra S R := hSR.toRingHom.toAlgebra
    this✝⁵ : IsDomain R := Function.Injective.isDomain (algebraMap R M) (NoZeroSMu …
    this✝⁴ : IsDomain S := Function.Injective.isDomain (algebraMap S L) (NoZeroSMu …
    this✝³ : Algebra R L := ((algebraMap S L).comp (algebraMap R S)).toAlgebra
    this✝² : IsScalarTower R S L
    this✝¹ : IsScalarTower S R L
    this✝ : NoZeroSMulDivisors R S
    this : Algebra.IsAlgebraic R L
    x : S
    ⊢ Eq ((IsAlgClosure.equivOfAlgebraic' R S L M) ((algebraMap S L) (hSR.symm (hS …
  -/
  show equivOfAlgebraic' R S L M (algebraMap R L (hSR x)) = _
  /-
    case a
    k : Type u
    inst✝¹⁴ : Field k
    K : Type u_1
    J : Type u_2
    R : Type u
    S : Type u_3
    L : Type v
    M : Type w
    inst✝¹³ : Field K
    inst✝¹² : Field J
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Field L
    inst✝⁸ : Field M
    inst✝⁷ : Algebra R M
    inst✝⁶ : NoZeroSMulDivisors R M
    inst✝⁵ : IsAlgClosure R M
    inst✝⁴ : Algebra K M
    inst✝³ : IsAlgClosure K M
    inst✝² : Algebra S L
    inst✝¹ : NoZeroSMulDivisors S L
    inst✝ : IsAlgClosure S L
    hSR : RingEquiv S R
    this✝⁷ : Algebra R S := hSR.symm.toRingHom.toAlgebra
    this✝⁶ : Algebra S R := hSR.toRingHom.toAlgebra
    this✝⁵ : IsDomain R := Function.Injective.isDomain (algebraMap R M) (NoZeroSMu …
    this✝⁴ : IsDomain S := Function.Injective.isDomain (algebraMap S L) (NoZeroSMu …
    this✝³ : Algebra R L := ((algebraMap S L).comp (algebraMap R S)).toAlgebra
    this✝² : IsScalarTower R S L
    this✝¹ : IsScalarTower S R L
    this✝ : NoZeroSMulDivisors R S
    this : Algebra.IsAlgebraic R L
    x : S
    ⊢ Eq ((IsAlgClosure.equivOfAlgebraic' R S L M) ((algebraMap R L) (hSR x))) ((a …
  -/
  rw [AlgEquiv.commutes]
  /-
    🎉 no goals
  -/


/-- Algebraic closure of isomorphic fields are isomorphic -/
noncomputable def equivOfEquiv (hSR : S ≃+* R) : L ≃+* M :=
  equivOfEquivAux L M hSR


@[simp]
theorem equivOfEquiv_comp_algebraMap (hSR : S ≃+* R) :
    (↑(equivOfEquiv L M hSR) : L →+* M).comp (algebraMap S L) = (algebraMap R M).comp hSR :=
  (equivOfEquivAux L M hSR).2


@[simp]
theorem equivOfEquiv_algebraMap (hSR : S ≃+* R) (s : S) :
    equivOfEquiv L M hSR (algebraMap S L s) = algebraMap R M (hSR s) :=
  RingHom.ext_iff.1 (equivOfEquiv_comp_algebraMap L M hSR) s


@[simp]
theorem equivOfEquiv_symm_algebraMap (hSR : S ≃+* R) (r : R) :
    (equivOfEquiv L M hSR).symm (algebraMap R M r) = algebraMap S L (hSR.symm r) :=
                                       /-
                                         R : Type u
                                         S : Type u_3
                                         L : Type v
                                         M : Type w
                                         inst✝⁹ : CommRing R
                                         inst✝⁸ : CommRing S
                                         inst✝⁷ : Field L
                                         inst✝⁶ : Field M
                                         inst✝⁵ : Algebra R M
                                         inst✝⁴ : NoZeroSMulDivisors R M
                                         inst✝³ : IsAlgClosure R M
                                         inst✝² : Algebra S L
                                         inst✝¹ : NoZeroSMulDivisors S L
                                         inst✝ : IsAlgClosure S L
                                         hSR : RingEquiv S R
                                         r : R
                                         ⊢ Eq ((IsAlgClosure.equivOfEquiv L M hSR) ((IsAlgClosure.equivOfEquiv L M hSR) …
                                       -/
  (equivOfEquiv L M hSR).injective (by simp)
                                       /-
                                         🎉 no goals
                                       -/


@[simp]
theorem equivOfEquiv_symm_comp_algebraMap (hSR : S ≃+* R) :
    ((equivOfEquiv L M hSR).symm : M →+* L).comp (algebraMap R M) =
      (algebraMap S L).comp hSR.symm :=
  RingHom.ext_iff.2 (equivOfEquiv_symm_algebraMap L M hSR)


/-- Let `A` be an algebraically closed field and let `x ∈ K`, with `K/F` an algebraic extension
  of fields. Then the images of `x` by the `F`-algebra morphisms from `K` to `A` are exactly
  the roots in `A` of the minimal polynomial of `x` over `F`. -/
theorem Algebra.IsAlgebraic.range_eval_eq_rootSet_minpoly [IsAlgClosed A] (x : K) :
    (Set.range fun ψ : K →ₐ[F] A ↦ ψ x) = (minpoly F x).rootSet A :=
  range_eval_eq_rootSet_minpoly_of_splits A (fun _ ↦ IsAlgClosed.splits_codomain _) x


/-- All `F`-embeddings of a field `K` into another field `A` factor through any intermediate
field of `A/F` in which the minimal polynomial of elements of `K` splits. -/
@[simps]
def IntermediateField.algHomEquivAlgHomOfSplits (L : IntermediateField F A)
    (hL : ∀ x : K, (minpoly F x).Splits (algebraMap F L)) :
    (K →ₐ[F] L) ≃ (K →ₐ[F] A) where
  toFun := L.val.comp
  invFun f := f.codRestrict _ fun x ↦
    ((Algebra.IsIntegral.isIntegral x).map f).mem_intermediateField_of_minpoly_splits <| by
      /-
        k : Type u
        inst✝⁶ : Field k
        F : Type u_1
        K : Type u_2
        A : Type u_3
        inst✝⁵ : Field F
        inst✝⁴ : Field K
        inst✝³ : Field A
        inst✝² : Algebra F K
        inst✝¹ : Algebra F A
        inst✝ : Algebra.IsAlgebraic F K
        L : IntermediateField F A
        hL : ∀ (x : K), Polynomial.Splits (algebraMap F (Subtype fun x => Membership.m …
        f : AlgHom F K A
        x : K
        ⊢ Polynomial.Splits (algebraMap F (Subtype fun x => Membership.mem L x)) (minp …
      -/
      rw [minpoly.algHom_eq f f.injective]; exact hL x
                                            /-
                                              🎉 no goals
                                            -/
  left_inv _ := rfl
                    /-
                      k : Type u
                      inst✝⁶ : Field k
                      F : Type u_1
                      K : Type u_2
                      A : Type u_3
                      inst✝⁵ : Field F
                      inst✝⁴ : Field K
                      inst✝³ : Field A
                      inst✝² : Algebra F K
                      inst✝¹ : Algebra F A
                      inst✝ : Algebra.IsAlgebraic F K
                      L : IntermediateField F A
                      hL : ∀ (x : K), Polynomial.Splits (algebraMap F (Subtype fun x => Membership.m …
                      x✝ : AlgHom F K A
                      ⊢ Eq (L.val.comp ((fun f => f.codRestrict L.toSubalgebra ⋯) x✝)) x✝
                    -/
  right_inv _ := by rfl
                    /-
                      🎉 no goals
                    -/


theorem IntermediateField.algHomEquivAlgHomOfSplits_apply_apply (L : IntermediateField F A)
    (hL : ∀ x : K, (minpoly F x).Splits (algebraMap F L)) (f : K →ₐ[F] L) (x : K) :
    algHomEquivAlgHomOfSplits A L hL f x = algebraMap L A (f x) := rfl


/-- All `F`-embeddings of a field `K` into another field `A` factor through any subextension
of `A/F` in which the minimal polynomial of elements of `K` splits. -/
noncomputable def Algebra.IsAlgebraic.algHomEquivAlgHomOfSplits (L : Type*) [Field L]
    [Algebra F L] [Algebra L A] [IsScalarTower F L A]
    (hL : ∀ x : K, (minpoly F x).Splits (algebraMap F L)) :
    (K →ₐ[F] L) ≃ (K →ₐ[F] A) :=
  (AlgEquiv.refl.arrowCongr (AlgEquiv.ofInjectiveField (IsScalarTower.toAlgHom F L A))).trans <|
    IntermediateField.algHomEquivAlgHomOfSplits A (IsScalarTower.toAlgHom F L A).fieldRange
    fun x ↦ splits_of_algHom (hL x) (AlgHom.rangeRestrict _)


theorem Algebra.IsAlgebraic.algHomEquivAlgHomOfSplits_apply_apply (L : Type*) [Field L]
    [Algebra F L] [Algebra L A] [IsScalarTower F L A]
    (hL : ∀ x : K, (minpoly F x).Splits (algebraMap F L)) (f : K →ₐ[F] L) (x : K) :
    Algebra.IsAlgebraic.algHomEquivAlgHomOfSplits A L hL f x = algebraMap L A (f x) := rfl


/-- Over an algebraically closed field of characteristic zero a necessary and sufficient condition
for the set of roots of a nonzero polynomial `f` to be a subset of the set of roots of `g` is that
`f` divides `f.derivative * g`. Over an integral domain, this is a sufficient but not necessary
condition. See `isRoot_of_isRoot_of_dvd_derivative_mul` -/
theorem Polynomial.isRoot_of_isRoot_iff_dvd_derivative_mul {K : Type*} [Field K]
    [IsAlgClosed K] [CharZero K] {f g : K[X]} (hf0 : f ≠ 0) :
    (∀ x, IsRoot f x → IsRoot g x) ↔ f ∣ f.derivative * g := by
  /-
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : IsAlgClosed K
    inst✝ : CharZero K
    f g : Polynomial K
    hf0 : Ne f 0
    ⊢ Iff (∀ (x : K), f.IsRoot x → g.IsRoot x) (Dvd.dvd f (HMul.hMul (Polynomial.d …
  -/
  refine ⟨?_, isRoot_of_isRoot_of_dvd_derivative_mul hf0⟩
  /-
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : IsAlgClosed K
    inst✝ : CharZero K
    f g : Polynomial K
    hf0 : Ne f 0
    ⊢ (∀ (x : K), f.IsRoot x → g.IsRoot x) → Dvd.dvd f (HMul.hMul (Polynomial.deri …
  -/
  by_cases hg0 : g = 0
    /-
      case pos
      K : Type u_1
      inst✝² : Field K
      inst✝¹ : IsAlgClosed K
      inst✝ : CharZero K
      f g : Polynomial K
      hf0 : Ne f 0
      hg0 : Eq g 0
      ⊢ (∀ (x : K), f.IsRoot x → g.IsRoot x) → Dvd.dvd f (HMul.hMul (Polynomial.deri …
    -/
  · simp [hg0]
    /-
      🎉 no goals
    -/
  /-
    case neg
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : IsAlgClosed K
    inst✝ : CharZero K
    f g : Polynomial K
    hf0 : Ne f 0
    hg0 : Not (Eq g 0)
    ⊢ (∀ (x : K), f.IsRoot x → g.IsRoot x) → Dvd.dvd f (HMul.hMul (Polynomial.deri …
  -/
  by_cases hdf0 : derivative f = 0
    /-
      case pos
      K : Type u_1
      inst✝² : Field K
      inst✝¹ : IsAlgClosed K
      inst✝ : CharZero K
      f g : Polynomial K
      hf0 : Ne f 0
      hg0 : Not (Eq g 0)
      hdf0 : Eq (Polynomial.derivative f) 0
      ⊢ (∀ (x : K), f.IsRoot x → g.IsRoot x) → Dvd.dvd f (HMul.hMul (Polynomial.deri …
    -/
  · rw [eq_C_of_derivative_eq_zero hdf0]
    /-
      case pos
      K : Type u_1
      inst✝² : Field K
      inst✝¹ : IsAlgClosed K
      inst✝ : CharZero K
      f g : Polynomial K
      hf0 : Ne f 0
      hg0 : Not (Eq g 0)
      hdf0 : Eq (Polynomial.derivative f) 0
      ⊢ (∀ (x : K), (Polynomial.C (f.coeff 0)).IsRoot x → g.IsRoot x) → Dvd.dvd (Pol …
    -/
    simp only [eval_C, derivative_C, zero_mul, dvd_zero, implies_true]
    /-
      🎉 no goals
    -/
  /-
    case neg
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : IsAlgClosed K
    inst✝ : CharZero K
    f g : Polynomial K
    hf0 : Ne f 0
    hg0 : Not (Eq g 0)
    hdf0 : Not (Eq (Polynomial.derivative f) 0)
    ⊢ (∀ (x : K), f.IsRoot x → g.IsRoot x) → Dvd.dvd f (HMul.hMul (Polynomial.deri …
  -/
  have hdg :  f.derivative * g ≠ 0 := mul_ne_zero hdf0 hg0
  /-
    case neg
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : IsAlgClosed K
    inst✝ : CharZero K
    f g : Polynomial K
    hf0 : Ne f 0
    hg0 : Not (Eq g 0)
    hdf0 : Not (Eq (Polynomial.derivative f) 0)
    hdg : Ne (HMul.hMul (Polynomial.derivative f) g) 0
    ⊢ (∀ (x : K), f.IsRoot x → g.IsRoot x) → Dvd.dvd f (HMul.hMul (Polynomial.deri …
  -/
  classical rw [Splits.dvd_iff_roots_le_roots (IsAlgClosed.splits f) hf0 hdg, Multiset.le_iff_count]
  /-
    case neg
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : IsAlgClosed K
    inst✝ : CharZero K
    f g : Polynomial K
    hf0 : Ne f 0
    hg0 : Not (Eq g 0)
    hdf0 : Not (Eq (Polynomial.derivative f) 0)
    hdg : Ne (HMul.hMul (Polynomial.derivative f) g) 0
    ⊢ (∀ (x : K), f.IsRoot x → g.IsRoot x) → ∀ (a : K), LE.le (Multiset.count a f. …
  -/
  simp only [count_roots, rootMultiplicity_mul hdg]
  /-
    case neg
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : IsAlgClosed K
    inst✝ : CharZero K
    f g : Polynomial K
    hf0 : Ne f 0
    hg0 : Not (Eq g 0)
    hdf0 : Not (Eq (Polynomial.derivative f) 0)
    hdg : Ne (HMul.hMul (Polynomial.derivative f) g) 0
    ⊢ (∀ (x : K), f.IsRoot x → g.IsRoot x) → ∀ (a : K), LE.le (Polynomial.rootMult …
  -/
  refine forall_imp fun a => ?_
  /-
    case neg
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : IsAlgClosed K
    inst✝ : CharZero K
    f g : Polynomial K
    hf0 : Ne f 0
    hg0 : Not (Eq g 0)
    hdf0 : Not (Eq (Polynomial.derivative f) 0)
    hdg : Ne (HMul.hMul (Polynomial.derivative f) g) 0
    a : K
    ⊢ (f.IsRoot a → g.IsRoot a) → LE.le (Polynomial.rootMultiplicity a f) (HAdd.hA …
  -/
  by_cases haf : f.eval a = 0
    /-
      case pos
      K : Type u_1
      inst✝² : Field K
      inst✝¹ : IsAlgClosed K
      inst✝ : CharZero K
      f g : Polynomial K
      hf0 : Ne f 0
      hg0 : Not (Eq g 0)
      hdf0 : Not (Eq (Polynomial.derivative f) 0)
      hdg : Ne (HMul.hMul (Polynomial.derivative f) g) 0
      a : K
      haf : Eq (Polynomial.eval a f) 0
      ⊢ (f.IsRoot a → g.IsRoot a) → LE.le (Polynomial.rootMultiplicity a f) (HAdd.hA …
    -/
  · have h0 : 0 < f.rootMultiplicity a := (rootMultiplicity_pos hf0).2 haf
    /-
      case pos
      K : Type u_1
      inst✝² : Field K
      inst✝¹ : IsAlgClosed K
      inst✝ : CharZero K
      f g : Polynomial K
      hf0 : Ne f 0
      hg0 : Not (Eq g 0)
      hdf0 : Not (Eq (Polynomial.derivative f) 0)
      hdg : Ne (HMul.hMul (Polynomial.derivative f) g) 0
      a : K
      haf : Eq (Polynomial.eval a f) 0
      h0 : LT.lt 0 (Polynomial.rootMultiplicity a f)
      ⊢ (f.IsRoot a → g.IsRoot a) → LE.le (Polynomial.rootMultiplicity a f) (HAdd.hA …
    -/
    rw [derivative_rootMultiplicity_of_root haf]
    /-
      case pos
      K : Type u_1
      inst✝² : Field K
      inst✝¹ : IsAlgClosed K
      inst✝ : CharZero K
      f g : Polynomial K
      hf0 : Ne f 0
      hg0 : Not (Eq g 0)
      hdf0 : Not (Eq (Polynomial.derivative f) 0)
      hdg : Ne (HMul.hMul (Polynomial.derivative f) g) 0
      a : K
      haf : Eq (Polynomial.eval a f) 0
      h0 : LT.lt 0 (Polynomial.rootMultiplicity a f)
      ⊢ (f.IsRoot a → g.IsRoot a) → LE.le (Polynomial.rootMultiplicity a f) (HAdd.hA …
    -/
    intro h
    calc rootMultiplicity a f
        = rootMultiplicity a f - 1 + 1 := (Nat.sub_add_cancel (Nat.succ_le_iff.1 h0)).symm
      _ ≤ rootMultiplicity a f - 1 + rootMultiplicity a g := add_le_add le_rfl (Nat.succ_le_iff.1
        ((rootMultiplicity_pos hg0).2 (h haf)))
    /-
      case neg
      K : Type u_1
      inst✝² : Field K
      inst✝¹ : IsAlgClosed K
      inst✝ : CharZero K
      f g : Polynomial K
      hf0 : Ne f 0
      hg0 : Not (Eq g 0)
      hdf0 : Not (Eq (Polynomial.derivative f) 0)
      hdg : Ne (HMul.hMul (Polynomial.derivative f) g) 0
      a : K
      haf : Not (Eq (Polynomial.eval a f) 0)
      ⊢ (f.IsRoot a → g.IsRoot a) → LE.le (Polynomial.rootMultiplicity a f) (HAdd.hA …
    -/
  · simp [haf, rootMultiplicity_eq_zero haf]
    /-
      🎉 no goals
    -/

