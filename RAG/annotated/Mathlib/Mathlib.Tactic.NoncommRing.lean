lemma nat_lit_mul_eq_nsmul [n.AtLeastTwo] : no_index (OfNat.ofNat n) * r = OfNat.ofNat n • r := by
  /-
    R : Type u_1
    inst✝¹ : NonAssocSemiring R
    r : R
    n : Nat
    inst✝ : n.AtLeastTwo
    ⊢ Eq (HMul.hMul (OfNat.ofNat n) r) (HSMul.hSMul (OfNat.ofNat n) r)
  -/
  simp only [nsmul_eq_mul, Nat.cast_ofNat]
  /-
    🎉 no goals
  -/

lemma mul_nat_lit_eq_nsmul [n.AtLeastTwo] : r * no_index (OfNat.ofNat n) = OfNat.ofNat n • r := by
  /-
    R : Type u_1
    inst✝¹ : NonAssocSemiring R
    r : R
    n : Nat
    inst✝ : n.AtLeastTwo
    ⊢ Eq (HMul.hMul r (OfNat.ofNat n)) (HSMul.hSMul (OfNat.ofNat n) r)
  -/
  simp only [nsmul_eq_mul', Nat.cast_ofNat]
  /-
    🎉 no goals
  -/


/-- A tactic for simplifying identities in not-necessarily-commutative rings.

An example:
```lean
example {R : Type*} [Ring R] (a b c : R) : a * (b + c + c - b) = 2 * a * c := by
  noncomm_ring
```

You can use `noncomm_ring [h]` to also simplify using `h`.
-/
syntax (name := noncomm_ring) "noncomm_ring" optConfig (discharger)?
  (" [" ((simpStar <|> simpErase <|> simpLemma),*,?) "]")? : tactic


macro_rules
  | `(tactic| noncomm_ring $cfg:optConfig $[$disch]? $[[$rules,*]]?) => do
    let rules' := rules.getD ⟨#[]⟩
    let tac ← `(tactic|
      (first | simp $cfg:optConfig $(disch)? only [
          -- Expand everything out.
          add_mul, mul_add, sub_eq_add_neg,
          -- Right associate all products.
          mul_assoc,
          -- Expand powers to numerals.
          pow_one, pow_zero, pow_succ,
          -- Replace multiplication by numerals with `zsmul`.
          one_mul, mul_one, zero_mul, mul_zero,
          nat_lit_mul_eq_nsmul, mul_nat_lit_eq_nsmul,
          -- Pull `zsmul n` out the front so `abel` can see them.
          mul_smul_comm, smul_mul_assoc,
          -- Pull out negations.
          neg_mul, mul_neg,
          -- user-specified simp lemmas
          $rules',*] |
        fail "`noncomm_ring` simp lemmas don't apply; try `abel` instead") <;>
      first | abel1 | abel_nf)
    -- if a manual rewrite rule is provided, we repeat the tactic
    -- (since abel might simplify and allow the rewrite to apply again)
    if rules.isSome then `(tactic| repeat1 ($tac;)) else `(tactic| $tac)


