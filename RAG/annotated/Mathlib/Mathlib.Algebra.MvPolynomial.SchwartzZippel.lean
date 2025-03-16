local notation:70 s:70 " ^^ " n:71 => piFinset fun i : Fin n ↦ s i


set_option linter.unusedVariables false in
/-- The **Schwartz-Zippel lemma**

For a nonzero multivariable polynomial `p` over an integral domain, the probability that `p`
evaluates to zero at points drawn at random from a product of finite subsets `S i` of the integral
domain is bounded by the supremum of `∑ i, degᵢ s / #(S i)` ranging over monomials `s` of `p`. -/
lemma schwartz_zippel_sup_sum :
    ∀ {n} {p : MvPolynomial (Fin n) R} (hp : p ≠ 0) (S : Fin n → Finset R),
      #{x ∈ S ^^ n | eval x p = 0} / ∏ i, (#(S i) : ℚ≥0) ≤
        p.support.sup fun s ↦ ∑ i, (s i / #(S i) : ℚ≥0)
  | 0, p, hp, S => by
    -- Because `p` is a polynomial over zero variables, it is constant.
    /-
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : DecidableEq R
      p : MvPolynomial (Fin 0) R
      hp : Ne p 0
      S : Fin 0 → Finset R
      ⊢ LE.le (HDiv.hDiv (↑(Finset.filter (fun x => Eq ((MvPolynomial.eval x) p) 0)  …
    -/
    rw [p.eq_C_of_isEmpty] at *
    /-
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : DecidableEq R
      p : MvPolynomial (Fin 0) R
      hp : Ne (MvPolynomial.C (MvPolynomial.coeff 0 p)) 0
      S : Fin 0 → Finset R
      ⊢ LE.le (HDiv.hDiv (↑(Finset.filter (fun x => Eq ((MvPolynomial.eval x) (MvPol …
    -/
    simp [C_ne_zero.mp hp]
    /-
      🎉 no goals
    -/
    -- Now, assume that the theorem holds for all polynomials in `n` variables.
  | n + 1, p, hp, S => by
    -- We can consider `p` to be a polynomial over multivariable polynomials in one fewer variables.
    /-
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : DecidableEq R
      n : Nat
      p : MvPolynomial (Fin (HAdd.hAdd n 1)) R
      hp : Ne p 0
      S : Fin (HAdd.hAdd n 1) → Finset R
      ⊢ LE.le (HDiv.hDiv (↑(Finset.filter (fun x => Eq ((MvPolynomial.eval x) p) 0)  …
    -/
    set p' : Polynomial (MvPolynomial (Fin n) R) := finSuccEquiv R n p with hp'
    -- Since `p` is not identically zero, there is some `k` such that `pₖ` is not identically zero.
    -- WLOG `k` is the largest such.
    /-
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : DecidableEq R
      n : Nat
      p : MvPolynomial (Fin (HAdd.hAdd n 1)) R
      hp : Ne p 0
      S : Fin (HAdd.hAdd n 1) → Finset R
      p' : Polynomial (MvPolynomial (Fin n) R) := (MvPolynomial.finSuccEquiv R n) p
      hp' : Eq p' ((MvPolynomial.finSuccEquiv R n) p)
      ⊢ LE.le (HDiv.hDiv (↑(Finset.filter (fun x => Eq ((MvPolynomial.eval x) p) 0)  …
    -/
    set k := p'.natDegree with hk
    /-
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : DecidableEq R
      n : Nat
      p : MvPolynomial (Fin (HAdd.hAdd n 1)) R
      hp : Ne p 0
      S : Fin (HAdd.hAdd n 1) → Finset R
      p' : Polynomial (MvPolynomial (Fin n) R) := (MvPolynomial.finSuccEquiv R n) p
      hp' : Eq p' ((MvPolynomial.finSuccEquiv R n) p)
      k : Nat := p'.natDegree
      hk : Eq k p'.natDegree
      ⊢ LE.le (HDiv.hDiv (↑(Finset.filter (fun x => Eq ((MvPolynomial.eval x) p) 0)  …
    -/
    set pₖ := p'.leadingCoeff with hpₖ
    /-
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : DecidableEq R
      n : Nat
      p : MvPolynomial (Fin (HAdd.hAdd n 1)) R
      hp : Ne p 0
      S : Fin (HAdd.hAdd n 1) → Finset R
      p' : Polynomial (MvPolynomial (Fin n) R) := (MvPolynomial.finSuccEquiv R n) p
      hp' : Eq p' ((MvPolynomial.finSuccEquiv R n) p)
      k : Nat := p'.natDegree
      hk : Eq k p'.natDegree
      pₖ : MvPolynomial (Fin n) R := p'.leadingCoeff
      hpₖ : Eq pₖ p'.leadingCoeff
      ⊢ LE.le (HDiv.hDiv (↑(Finset.filter (fun x => Eq ((MvPolynomial.eval x) p) 0)  …
    -/
    have hp'₀ : p' ≠ 0 := EmbeddingLike.map_ne_zero_iff.2 hp
    /-
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : DecidableEq R
      n : Nat
      p : MvPolynomial (Fin (HAdd.hAdd n 1)) R
      hp : Ne p 0
      S : Fin (HAdd.hAdd n 1) → Finset R
      p' : Polynomial (MvPolynomial (Fin n) R) := (MvPolynomial.finSuccEquiv R n) p
      hp' : Eq p' ((MvPolynomial.finSuccEquiv R n) p)
      k : Nat := p'.natDegree
      hk : Eq k p'.natDegree
      pₖ : MvPolynomial (Fin n) R := p'.leadingCoeff
      hpₖ : Eq pₖ p'.leadingCoeff
      hp'₀ : Ne p' 0
      ⊢ LE.le (HDiv.hDiv (↑(Finset.filter (fun x => Eq ((MvPolynomial.eval x) p) 0)  …
    -/
    have hpₖ₀ : pₖ ≠ 0 := by simpa [pₖ, k]
    calc
      -- We split the set of possible zeros into a union of two cases.
      #{x ∈ S ^^ (n + 1) | eval x p = 0} / ∏ i, (#(S i) : ℚ≥0)
          -- In the first case, `pₖ` evaluates to `0`.
        = #{x ∈ S ^^ (n + 1) | eval x p = 0 ∧ eval (tail x) pₖ = 0} / ∏ i, (#(S i) : ℚ≥0)
          -- In the second case, `pₖ` does not evaluate to `0`.
          + #{x ∈ S ^^ (n + 1) | eval x p = 0 ∧ eval (tail x) pₖ ≠ 0} / ∏ i, (#(S i) : ℚ≥0) := by
        rw [← add_div, ← Nat.cast_add, ← card_union_add_card_inter, filter_union_right,
          ← filter_and]
        simp [← and_or_left, em, and_and_and_comm]
      _ ≤ (pₖ.support.sup fun s ↦ ∑ i, (s i / #(S i.succ) : ℚ≥0)) + p.degreeOf 0 / #(S 0) := ?_
      _ ≤ p.support.sup fun s ↦ ∑ i, (s i / #(S i) : ℚ≥0) := ?_
      /-
        case calc_1
        R : Type u_1
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        inst✝ : DecidableEq R
        n : Nat
        p : MvPolynomial (Fin (HAdd.hAdd n 1)) R
        hp : Ne p 0
        S : Fin (HAdd.hAdd n 1) → Finset R
        p' : Polynomial (MvPolynomial (Fin n) R) := (MvPolynomial.finSuccEquiv R n) p
        hp' : Eq p' ((MvPolynomial.finSuccEquiv R n) p)
        k : Nat := p'.natDegree
        hk : Eq k p'.natDegree
        pₖ : MvPolynomial (Fin n) R := p'.leadingCoeff
        hpₖ : Eq pₖ p'.leadingCoeff
        hp'₀ : Ne p' 0
        hpₖ₀ : Ne pₖ 0
        ⊢ LE.le (HAdd.hAdd (HDiv.hDiv (↑(Finset.filter (fun x => And (Eq ((MvPolynomia …
      -/
    · gcongr ?_ + ?_
      · -- We bound the size of the first set by induction
        calc
          #{x ∈ S ^^ (n + 1) | eval x p = 0 ∧ eval (tail x) pₖ = 0} / ∏ i, (#(S i) : ℚ≥0)
            ≤ #{x ∈ S ^^ (n + 1) | eval (tail x) pₖ = 0} / ∏ i, (#(S i) : ℚ≥0) := by
            gcongr; exact fun x hx ↦ hx.2
          _ = #(S 0) * #{xₜ ∈ tail S ^^ n | eval xₜ pₖ = 0}
              / (#(S 0) * (∏ i, #(S (.succ i)) : ℚ≥0)) := by
            rw [card_consEquiv_filter_piFinset S fun x ↦ eval x pₖ = 0, prod_univ_succ, tail_def]
            norm_cast
          _ ≤ #{xₜ ∈ tail S ^^ n | eval xₜ pₖ = 0} / ∏ i, (#(S (.succ i)) : ℚ≥0) :=
            mul_div_mul_left_le (by positivity)
          _ ≤ (pₖ.support.sup fun s ↦ ∑ i, (s i / #(S (.succ i)) : ℚ≥0)) :=
            schwartz_zippel_sup_sum hpₖ₀ _
      · -- We bound the second set by noting that if `x` is in it, then `x₀` is the root of
        -- the univariate polynomial`pₓ` obtained by evaluating each (multivariate polynomial)
        -- coefficient at `xₜ`. Since `pₓ` has degree `k`, there are at most `k` such `x₀` for
        -- each `xₜ`, which gives the result.
        calc
          #{x ∈ S ^^ (n + 1) | eval x p = 0 ∧ eval (tail x) pₖ ≠ 0} / ∏ i, (#(S i) : ℚ≥0)
            ≤ ↑(p.degreeOf 0 * ∏ i, #(S (.succ i))) / ∏ i, (#(S i) : ℚ≥0) := ?_
          _ = p.degreeOf 0 * (∏ i, #(S (.succ i))) / (#(S 0) * ∏ i, #(S (.succ i))) := by
            norm_cast; rw [prod_univ_succ]
          _ ≤ (p.degreeOf 0 / #(S 0) : ℚ≥0) := mul_div_mul_right_le (by positivity)
        /-
          case calc_1.h₂
          R : Type u_1
          inst✝² : CommRing R
          inst✝¹ : IsDomain R
          inst✝ : DecidableEq R
          n : Nat
          p : MvPolynomial (Fin (HAdd.hAdd n 1)) R
          hp : Ne p 0
          S : Fin (HAdd.hAdd n 1) → Finset R
          p' : Polynomial (MvPolynomial (Fin n) R) := (MvPolynomial.finSuccEquiv R n) p
          hp' : Eq p' ((MvPolynomial.finSuccEquiv R n) p)
          k : Nat := p'.natDegree
          hk : Eq k p'.natDegree
          pₖ : MvPolynomial (Fin n) R := p'.leadingCoeff
          hpₖ : Eq pₖ p'.leadingCoeff
          hp'₀ : Ne p' 0
          hpₖ₀ : Ne pₖ 0
          ⊢ LE.le (HDiv.hDiv (↑(Finset.filter (fun x => And (Eq ((MvPolynomial.eval x) p …
        -/
        gcongr
        calc
          #{x ∈ S ^^ (n + 1) | eval x p = 0 ∧ eval (tail x) pₖ ≠ 0}
            = #{x ∈ S ^^ (n + 1) | eval (tail x) pₖ ≠ 0 ∧ eval x p = 0} := by simp_rw [and_comm]
          _ = #({xₜ ∈ tail S ^^ n | eval xₜ pₖ ≠ 0}.biUnion fun xₜ ↦ image (fun x₀ ↦ (x₀, xₜ))
                {x₀ ∈ S 0 | eval (cons x₀ xₜ) p = 0}) := by
            rw [← filter_filter, filter_piFinset_eq_map_consEquiv S (fun r ↦ eval r pₖ ≠ 0),
              filter_map, card_map, product_eq_biUnion_right, filter_biUnion]
            simp [Function.comp_def, filter_image, filter_filter]
            rfl
          _ ≤ ∑ xₜ ∈ tail S ^^ n with eval xₜ pₖ ≠ 0,
                #(image (fun x₀ ↦ (x₀, xₜ)) {x₀ ∈ S 0 | eval (cons x₀ xₜ) p = 0}) :=
            card_biUnion_le
          _ ≤ ∑ xₜ ∈ tail S ^^ n with eval xₜ pₖ ≠ 0, #{x₀ ∈ S 0 | eval (cons x₀ xₜ) p = 0} := by
            gcongr; exact card_image_le
          _ ≤ ∑ xₜ ∈ tail S ^^ n with eval xₜ pₖ ≠ 0, p.degreeOf 0 := ?_
          _ ≤ ∑ _xₜ ∈ tail S ^^ n, p.degreeOf 0 := by gcongr; exact filter_subset ..
          _ = p.degreeOf 0 * ∏ i, #(S (.succ i)) := by simp [mul_comm, tail]
        /-
          case calc_1.h₂.hab.h
          R : Type u_1
          inst✝² : CommRing R
          inst✝¹ : IsDomain R
          inst✝ : DecidableEq R
          n : Nat
          p : MvPolynomial (Fin (HAdd.hAdd n 1)) R
          hp : Ne p 0
          S : Fin (HAdd.hAdd n 1) → Finset R
          p' : Polynomial (MvPolynomial (Fin n) R) := (MvPolynomial.finSuccEquiv R n) p
          hp' : Eq p' ((MvPolynomial.finSuccEquiv R n) p)
          k : Nat := p'.natDegree
          hk : Eq k p'.natDegree
          pₖ : MvPolynomial (Fin n) R := p'.leadingCoeff
          hpₖ : Eq pₖ p'.leadingCoeff
          hp'₀ : Ne p' 0
          hpₖ₀ : Ne pₖ 0
          ⊢ LE.le ((Finset.filter (fun xₜ => Ne ((MvPolynomial.eval xₜ) pₖ) 0) (Fintype. …
        -/
        gcongr with xₜ hxₜ
        /-
          case calc_1.h₂.hab.h.h
          R : Type u_1
          inst✝² : CommRing R
          inst✝¹ : IsDomain R
          inst✝ : DecidableEq R
          n : Nat
          p : MvPolynomial (Fin (HAdd.hAdd n 1)) R
          hp : Ne p 0
          S : Fin (HAdd.hAdd n 1) → Finset R
          p' : Polynomial (MvPolynomial (Fin n) R) := (MvPolynomial.finSuccEquiv R n) p
          hp' : Eq p' ((MvPolynomial.finSuccEquiv R n) p)
          k : Nat := p'.natDegree
          hk : Eq k p'.natDegree
          pₖ : MvPolynomial (Fin n) R := p'.leadingCoeff
          hpₖ : Eq pₖ p'.leadingCoeff
          hp'₀ : Ne p' 0
          hpₖ₀ : Ne pₖ 0
          xₜ : Fin n → R
          hxₜ : Membership.mem (Finset.filter (fun xₜ => Ne ((MvPolynomial.eval xₜ) pₖ)  …
          ⊢ LE.le (Finset.filter (fun x₀ => Eq ((MvPolynomial.eval (Fin.cons x₀ xₜ)) p)  …
        -/
        set pₓ := p'.map (eval xₜ) with hpₓ
        have hpₓdeg : pₓ.natDegree = k := by
          rw [hpₓ, hk, Polynomial.natDegree_map_of_leadingCoeff_ne_zero _ (mem_filter.1 hxₜ).2]
        have hpₓ₀ : pₓ ≠ 0 := fun h ↦ (mem_filter.1 hxₜ).2 <| by
          rw [hpₖ, Polynomial.leadingCoeff, ← hk, ← hpₓdeg, h, Polynomial.natDegree_zero,
            ← Polynomial.coeff_map, ← hpₓ, h, Polynomial.coeff_zero]
        calc
          #{x₀ ∈ S 0 | eval (cons x₀ xₜ) p = 0} ≤ #pₓ.roots.toFinset := by
            gcongr
            simp (config := { contextual := true }) [subset_iff, eval_eq_eval_mv_eval', pₓ, hpₓ₀]
          _ ≤ Multiset.card pₓ.roots := pₓ.roots.toFinset_card_le
          _ ≤ pₓ.natDegree := pₓ.card_roots'
          _ = k := hpₓdeg
          _ ≤ p.degreeOf 0 := by
            have :
              (ofLex (AddMonoidAlgebra.supDegree toLex p'.leadingCoeff)).cons k ∈ p.support := by
              rwa [← support_coeff_finSuccEquiv, mem_support_iff, ← hp', hk,
                ← Polynomial.leadingCoeff, ← hpₖ, ← leadingCoeff_toLex,
                AddMonoidAlgebra.leadingCoeff_ne_zero toLex.injective]
            simpa using monomial_le_degreeOf 0 this
      /-
        case calc_2
        R : Type u_1
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        inst✝ : DecidableEq R
        n : Nat
        p : MvPolynomial (Fin (HAdd.hAdd n 1)) R
        hp : Ne p 0
        S : Fin (HAdd.hAdd n 1) → Finset R
        p' : Polynomial (MvPolynomial (Fin n) R) := (MvPolynomial.finSuccEquiv R n) p
        hp' : Eq p' ((MvPolynomial.finSuccEquiv R n) p)
        k : Nat := p'.natDegree
        hk : Eq k p'.natDegree
        pₖ : MvPolynomial (Fin n) R := p'.leadingCoeff
        hpₖ : Eq pₖ p'.leadingCoeff
        hp'₀ : Ne p' 0
        hpₖ₀ : Ne pₖ 0
        ⊢ LE.le (HAdd.hAdd (pₖ.support.sup fun s => Finset.univ.sum fun i => HDiv.hDiv …
      -/
    · rw [Finset.sup_add (support_nonempty.mpr hpₖ₀)]
      /-
        case calc_2
        R : Type u_1
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        inst✝ : DecidableEq R
        n : Nat
        p : MvPolynomial (Fin (HAdd.hAdd n 1)) R
        hp : Ne p 0
        S : Fin (HAdd.hAdd n 1) → Finset R
        p' : Polynomial (MvPolynomial (Fin n) R) := (MvPolynomial.finSuccEquiv R n) p
        hp' : Eq p' ((MvPolynomial.finSuccEquiv R n) p)
        k : Nat := p'.natDegree
        hk : Eq k p'.natDegree
        pₖ : MvPolynomial (Fin n) R := p'.leadingCoeff
        hpₖ : Eq pₖ p'.leadingCoeff
        hp'₀ : Ne p' 0
        hpₖ₀ : Ne pₖ 0
        ⊢ LE.le (pₖ.support.sup fun i => HAdd.hAdd (Finset.univ.sum fun i_1 => HDiv.hD …
      -/
      apply Finset.sup_le
      /-
        case calc_2.a
        R : Type u_1
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        inst✝ : DecidableEq R
        n : Nat
        p : MvPolynomial (Fin (HAdd.hAdd n 1)) R
        hp : Ne p 0
        S : Fin (HAdd.hAdd n 1) → Finset R
        p' : Polynomial (MvPolynomial (Fin n) R) := (MvPolynomial.finSuccEquiv R n) p
        hp' : Eq p' ((MvPolynomial.finSuccEquiv R n) p)
        k : Nat := p'.natDegree
        hk : Eq k p'.natDegree
        pₖ : MvPolynomial (Fin n) R := p'.leadingCoeff
        hpₖ : Eq pₖ p'.leadingCoeff
        hp'₀ : Ne p' 0
        hpₖ₀ : Ne pₖ 0
        ⊢ ∀ (b : Finsupp (Fin n) Nat), Membership.mem pₖ.support b → LE.le (HAdd.hAdd  …
      -/
      rintro i hi
      /-
        case calc_2.a
        R : Type u_1
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        inst✝ : DecidableEq R
        n : Nat
        p : MvPolynomial (Fin (HAdd.hAdd n 1)) R
        hp : Ne p 0
        S : Fin (HAdd.hAdd n 1) → Finset R
        p' : Polynomial (MvPolynomial (Fin n) R) := (MvPolynomial.finSuccEquiv R n) p
        hp' : Eq p' ((MvPolynomial.finSuccEquiv R n) p)
        k : Nat := p'.natDegree
        hk : Eq k p'.natDegree
        pₖ : MvPolynomial (Fin n) R := p'.leadingCoeff
        hpₖ : Eq pₖ p'.leadingCoeff
        hp'₀ : Ne p' 0
        hpₖ₀ : Ne pₖ 0
        i : Finsupp (Fin n) Nat
        hi : Membership.mem pₖ.support i
        ⊢ LE.le (HAdd.hAdd (Finset.univ.sum fun i_1 => HDiv.hDiv ↑(i i_1) ↑(S i_1.succ …
      -/
      refine le_sup_of_le (mem_support_coeff_finSuccEquiv.mp hi) ?_
      /-
        case calc_2.a
        R : Type u_1
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        inst✝ : DecidableEq R
        n : Nat
        p : MvPolynomial (Fin (HAdd.hAdd n 1)) R
        hp : Ne p 0
        S : Fin (HAdd.hAdd n 1) → Finset R
        p' : Polynomial (MvPolynomial (Fin n) R) := (MvPolynomial.finSuccEquiv R n) p
        hp' : Eq p' ((MvPolynomial.finSuccEquiv R n) p)
        k : Nat := p'.natDegree
        hk : Eq k p'.natDegree
        pₖ : MvPolynomial (Fin n) R := p'.leadingCoeff
        hpₖ : Eq pₖ p'.leadingCoeff
        hp'₀ : Ne p' 0
        hpₖ₀ : Ne pₖ 0
        i : Finsupp (Fin n) Nat
        hi : Membership.mem pₖ.support i
        ⊢ LE.le (HAdd.hAdd (Finset.univ.sum fun i_1 => HDiv.hDiv ↑(i i_1) ↑(S i_1.succ …
      -/
      rw [Fin.sum_univ_succ, add_comm]
      /-
        case calc_2.a
        R : Type u_1
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        inst✝ : DecidableEq R
        n : Nat
        p : MvPolynomial (Fin (HAdd.hAdd n 1)) R
        hp : Ne p 0
        S : Fin (HAdd.hAdd n 1) → Finset R
        p' : Polynomial (MvPolynomial (Fin n) R) := (MvPolynomial.finSuccEquiv R n) p
        hp' : Eq p' ((MvPolynomial.finSuccEquiv R n) p)
        k : Nat := p'.natDegree
        hk : Eq k p'.natDegree
        pₖ : MvPolynomial (Fin n) R := p'.leadingCoeff
        hpₖ : Eq pₖ p'.leadingCoeff
        hp'₀ : Ne p' 0
        hpₖ₀ : Ne pₖ 0
        i : Finsupp (Fin n) Nat
        hi : Membership.mem pₖ.support i
        ⊢ LE.le (HAdd.hAdd (HDiv.hDiv ↑(MvPolynomial.degreeOf 0 p) ↑(S 0).card) (Finse …
      -/
      dsimp
      /-
        case calc_2.a
        R : Type u_1
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        inst✝ : DecidableEq R
        n : Nat
        p : MvPolynomial (Fin (HAdd.hAdd n 1)) R
        hp : Ne p 0
        S : Fin (HAdd.hAdd n 1) → Finset R
        p' : Polynomial (MvPolynomial (Fin n) R) := (MvPolynomial.finSuccEquiv R n) p
        hp' : Eq p' ((MvPolynomial.finSuccEquiv R n) p)
        k : Nat := p'.natDegree
        hk : Eq k p'.natDegree
        pₖ : MvPolynomial (Fin n) R := p'.leadingCoeff
        hpₖ : Eq pₖ p'.leadingCoeff
        hp'₀ : Ne p' 0
        hpₖ₀ : Ne pₖ 0
        i : Finsupp (Fin n) Nat
        hi : Membership.mem pₖ.support i
        ⊢ LE.le (HAdd.hAdd (HDiv.hDiv ↑(MvPolynomial.degreeOf 0 p) ↑(S 0).card) (Finse …
      -/
      gcongr
      /-
        case calc_2.a.bc.hab.h
        R : Type u_1
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        inst✝ : DecidableEq R
        n : Nat
        p : MvPolynomial (Fin (HAdd.hAdd n 1)) R
        hp : Ne p 0
        S : Fin (HAdd.hAdd n 1) → Finset R
        p' : Polynomial (MvPolynomial (Fin n) R) := (MvPolynomial.finSuccEquiv R n) p
        hp' : Eq p' ((MvPolynomial.finSuccEquiv R n) p)
        k : Nat := p'.natDegree
        hk : Eq k p'.natDegree
        pₖ : MvPolynomial (Fin n) R := p'.leadingCoeff
        hpₖ : Eq pₖ p'.leadingCoeff
        hp'₀ : Ne p' 0
        hpₖ₀ : Ne pₖ 0
        i : Finsupp (Fin n) Nat
        hi : Membership.mem pₖ.support i
        ⊢ LE.le (MvPolynomial.degreeOf 0 p) p'.natDegree
      -/
      simp [k, natDegree_finSuccEquiv, p']
      /-
        🎉 no goals
      -/


/-- The **Schwartz-Zippel lemma**

For a nonzero multivariable polynomial `p` over an integral domain, the probability that `p`
evaluates to zero at points drawn at random from a product of finite subsets `S i` of the integral
domain is bounded by the sum of `degᵢ p / #(S i)`. -/
lemma schwartz_zippel_sum_degreeOf {n} {p : MvPolynomial (Fin n) R} (hp : p ≠ 0)
    (S : Fin n → Finset R) :
    #{x ∈ S ^^ n | eval x p = 0} / ∏ i, (#(S i) : ℚ≥0) ≤ ∑ i, (p.degreeOf i / #(S i) : ℚ≥0) := by
  calc
    _ ≤ p.support.sup fun s ↦ ∑ i, (s i / #(S i) : ℚ≥0) := schwartz_zippel_sup_sum hp S
    _ ≤ ∑ i, (p.degreeOf i / #(S i) : ℚ≥0) := Finset.sup_le fun s hs ↦ by
      gcongr with i; exact monomial_le_degreeOf i hs


/-- The **Schwartz-Zippel lemma**

For a nonzero multivariable polynomial `p` over an integral domain, the probability that `p`
evaluates to zero at points drawn at random from some finite subset `S` of the integral domain is
bounded by the degree of `p` over `#S`. This version presents this lemma in terms of `Finset`. -/
lemma schwartz_zippel_totalDegree {n} {p : MvPolynomial (Fin n) R} (hp : p ≠ 0) (S : Finset R) :
    #{f ∈ piFinset fun _ ↦ S | eval f p = 0} / (#S ^ n : ℚ≥0) ≤ p.totalDegree / #S :=
  calc
                                                                                 /-
                                                                                   R : Type u_1
                                                                                   inst✝² : CommRing R
                                                                                   inst✝¹ : IsDomain R
                                                                                   inst✝ : DecidableEq R
                                                                                   n : Nat
                                                                                   p : MvPolynomial (Fin n) R
                                                                                   hp : Ne p 0
                                                                                   S : Finset R
                                                                                   ⊢ Eq (HDiv.hDiv (↑(Finset.filter (fun f => Eq ((MvPolynomial.eval f) p) 0) (Fi …
                                                                                 -/
    _ = #{f ∈ piFinset fun _ ↦ S | eval f p = 0} / (∏ i : Fin n, #S : ℚ≥0) := by simp
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
    _ ≤ p.support.sup fun s ↦ ∑ i, (s i / #S : ℚ≥0) := schwartz_zippel_sup_sum hp _
    _ = p.totalDegree / #S := by
      /-
        R : Type u_1
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        inst✝ : DecidableEq R
        n : Nat
        p : MvPolynomial (Fin n) R
        hp : Ne p 0
        S : Finset R
        ⊢ Eq (p.support.sup fun s => Finset.univ.sum fun i => HDiv.hDiv ↑(s i) ↑S.card …
      -/
      obtain rfl | hs := S.eq_empty_or_nonempty
        /-
          case inl
          R : Type u_1
          inst✝² : CommRing R
          inst✝¹ : IsDomain R
          inst✝ : DecidableEq R
          n : Nat
          p : MvPolynomial (Fin n) R
          hp : Ne p 0
          ⊢ Eq (p.support.sup fun s => Finset.univ.sum fun i => HDiv.hDiv ↑(s i) ↑EmptyC …
        -/
      · simp
        /-
          case inl
          R : Type u_1
          inst✝² : CommRing R
          inst✝¹ : IsDomain R
          inst✝ : DecidableEq R
          n : Nat
          p : MvPolynomial (Fin n) R
          hp : Ne p 0
          ⊢ Eq (p.support.sup fun s => 0) 0
        -/
        simp only [← _root_.bot_eq_zero, sup_bot]
        /-
          🎉 no goals
        -/
      /-
        case inr
        R : Type u_1
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        inst✝ : DecidableEq R
        n : Nat
        p : MvPolynomial (Fin n) R
        hp : Ne p 0
        S : Finset R
        hs : S.Nonempty
        ⊢ Eq (p.support.sup fun s => Finset.univ.sum fun i => HDiv.hDiv ↑(s i) ↑S.card …
      -/
      simp_rw [totalDegree, Nat.cast_finsetSup]
      /-
        case inr
        R : Type u_1
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        inst✝ : DecidableEq R
        n : Nat
        p : MvPolynomial (Fin n) R
        hp : Ne p 0
        S : Finset R
        hs : S.Nonempty
        ⊢ Eq (p.support.sup fun s => Finset.univ.sum fun i => HDiv.hDiv ↑(s i) ↑S.card …
      -/
      rw [sup_div₀ (ha := show 0 < (#S : ℚ≥0) by positivity)]
      /-
        case inr
        R : Type u_1
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        inst✝ : DecidableEq R
        n : Nat
        p : MvPolynomial (Fin n) R
        hp : Ne p 0
        S : Finset R
        hs : S.Nonempty
        ⊢ Eq (p.support.sup fun s => Finset.univ.sum fun i => HDiv.hDiv ↑(s i) ↑S.card …
      -/
      simp [← sum_div, Finsupp.sum_fintype]
      /-
        🎉 no goals
      -/


