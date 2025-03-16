local notation a " /ℚ " q => (q : ℚ≥0)⁻¹ • a


/-- Average of a function over a finset. If the finset is empty, this is equal to zero. -/
def Finset.expect [AddCommMonoid M] [Module ℚ≥0 M] (s : Finset ι) (f : ι → M) : M :=
  (#s : ℚ≥0)⁻¹ • ∑ i ∈ s, f i


/--
* `𝔼 i ∈ s, f i` is notation for `Finset.expect s f`. It is the expectation of `f i` where `i`
  ranges over the finite set `s` (either a `Finset` or a `Set` with a `Fintype` instance).
* `𝔼 i, f i` is notation for `Finset.expect Finset.univ f`. It is the expectation of `f i` where `i`
  ranges over the finite domain of `f`.
* `𝔼 i ∈ s with p i, f i` is notation for `Finset.expect (Finset.filter p s) f`.
* `𝔼 (i ∈ s) (j ∈ t), f i j` is notation for `Finset.expect (s ×ˢ t) (fun ⟨i, j⟩ ↦ f i j)`.

These support destructuring, for example `𝔼 ⟨i, j⟩ ∈ s ×ˢ t, f i j`.

Notation: `"𝔼" bigOpBinders* ("with" term)? "," term` -/
scoped syntax (name := bigexpect) "𝔼 " bigOpBinders ("with " term)? ", " term:67 : term


scoped macro_rules (kind := bigexpect)
  | `(𝔼 $bs:bigOpBinders $[with $p?]?, $v) => do
    let processed ← processBigOpBinders bs
    let i ← bigOpBindersPattern processed
    let s ← bigOpBindersProd processed
    match p? with
    | some p => `(Finset.expect (Finset.filter (fun $i ↦ $p) $s) (fun $i ↦ $v))
    | none => `(Finset.expect $s (fun $i ↦ $v))


/-- Delaborator for `Finset.expect`. The `pp.piBinderTypes` option controls whether
to show the domain type when the expect is over `Finset.univ`. -/
@[scoped app_delab Finset.expect] def delabFinsetExpect : Delab :=
  whenPPOption getPPNotation <| withOverApp 6 <| do
  let #[_, _, _, _, s, f] := (← getExpr).getAppArgs | failure
  guard <| f.isLambda
  let ppDomain ← getPPOption getPPPiBinderTypes
  let (i, body) ← withAppArg <| withBindingBodyUnusedName fun i => do
    return (i, ← delab)
  if s.isAppOfArity ``Finset.univ 2 then
    let binder ←
      if ppDomain then
        let ty ← withNaryArg 0 delab
        `(bigOpBinder| $(.mk i):ident : $ty)
      else
        `(bigOpBinder| $(.mk i):ident)
    `(𝔼 $binder:bigOpBinder, $body)
  else
    let ss ← withNaryArg 4 <| delab
    `(𝔼 $(.mk i):ident ∈ $ss, $body)


lemma expect_univ [Fintype ι] : 𝔼 i, f i = (∑ i, f i) /ℚ Fintype.card ι := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝² : AddCommMonoid M
    inst✝¹ : Module NNRat M
    f : ι → M
    inst✝ : Fintype ι
    ⊢ Eq (Finset.univ.expect fun i => f i) (HSMul.hSMul (Inv.inv ↑(Fintype.card ι) …
  -/
  rw [expect, card_univ]
  /-
    🎉 no goals
  -/


                                                                /-
                                                                  ι : Type u_1
                                                                  M : Type u_3
                                                                  inst✝¹ : AddCommMonoid M
                                                                  inst✝ : Module NNRat M
                                                                  f : ι → M
                                                                  ⊢ Eq (EmptyCollection.emptyCollection.expect fun i => f i) 0
                                                                -/
@[simp] lemma expect_empty (f : ι → M) : 𝔼 i ∈ ∅, f i = 0 := by simp [expect]
                                                                /-
                                                                  🎉 no goals
                                                                -/

                                                                                /-
                                                                                  ι : Type u_1
                                                                                  M : Type u_3
                                                                                  inst✝¹ : AddCommMonoid M
                                                                                  inst✝ : Module NNRat M
                                                                                  f : ι → M
                                                                                  i : ι
                                                                                  ⊢ Eq ((Singleton.singleton i).expect fun j => f j) (f i)
                                                                                -/
@[simp] lemma expect_singleton (f : ι → M) (i : ι) : 𝔼 j ∈ {i}, f j = f i := by simp [expect]
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/

                                                                             /-
                                                                               ι : Type u_1
                                                                               M : Type u_3
                                                                               inst✝¹ : AddCommMonoid M
                                                                               inst✝ : Module NNRat M
                                                                               s : Finset ι
                                                                               ⊢ Eq (s.expect fun _i => 0) 0
                                                                             -/
@[simp] lemma expect_const_zero (s : Finset ι) : 𝔼 _i ∈ s, (0 : M) = 0 := by simp [expect]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


@[congr]
lemma expect_congr {t : Finset ι} (hst : s = t) (h : ∀ i ∈ t, f i = g i) :
                                      /-
                                        ι : Type u_1
                                        M : Type u_3
                                        inst✝¹ : AddCommMonoid M
                                        inst✝ : Module NNRat M
                                        s : Finset ι
                                        f g : ι → M
                                        t : Finset ι
                                        hst : Eq s t
                                        h : ∀ (i : ι), Membership.mem t i → Eq (f i) (g i)
                                        ⊢ Eq (s.expect fun i => f i) (t.expect fun i => g i)
                                      -/
    𝔼 i ∈ s, f i = 𝔼 i ∈ t, g i := by rw [expect, expect, sum_congr hst h, hst]
                                      /-
                                        🎉 no goals
                                      -/


lemma expectWith_congr (hst : s = t) (hpq : ∀ i ∈ t, p i ↔ q i) (h : ∀ i ∈ t, q i → f i = g i) :
    𝔼 i ∈ s with p i, f i = 𝔼 i ∈ t with q i, g i :=
                   /-
                     ι : Type u_1
                     M : Type u_3
                     inst✝³ : AddCommMonoid M
                     inst✝² : Module NNRat M
                     s t : Finset ι
                     f g : ι → M
                     p q : ι → Prop
                     inst✝¹ : DecidablePred p
                     inst✝ : DecidablePred q
                     hst : Eq s t
                     hpq : ∀ (i : ι), Membership.mem t i → Iff (p i) (q i)
                     h : ∀ (i : ι), Membership.mem t i → q i → Eq (f i) (g i)
                     ⊢ Eq (Finset.filter (fun i => p i) s) (Finset.filter (fun i => q i) t)
                   -/
                   /-
                     🎉 no goals
                   -/
  expect_congr (by rw [hst, filter_inj'.2 hpq]) <| by simpa using h
                                                      /-
                                                        🎉 no goals
                                                      -/


lemma expect_sum_comm (s : Finset ι) (t : Finset κ) (f : ι → κ → M) :
    𝔼 i ∈ s, ∑ j ∈ t, f i j = ∑ j ∈ t, 𝔼 i ∈ s, f i j := by
  /-
    ι : Type u_1
    κ : Type u_2
    M : Type u_3
    inst✝¹ : AddCommMonoid M
    inst✝ : Module NNRat M
    s : Finset ι
    t : Finset κ
    f : ι → κ → M
    ⊢ Eq (s.expect fun i => t.sum fun j => f i j) (t.sum fun j => s.expect fun i = …
  -/
  simpa only [expect, smul_sum] using sum_comm
  /-
    🎉 no goals
  -/


lemma expect_comm (s : Finset ι) (t : Finset κ) (f : ι → κ → M) :
    𝔼 i ∈ s, 𝔼 j ∈ t, f i j = 𝔼 j ∈ t, 𝔼 i ∈ s, f i j := by
  /-
    ι : Type u_1
    κ : Type u_2
    M : Type u_3
    inst✝¹ : AddCommMonoid M
    inst✝ : Module NNRat M
    s : Finset ι
    t : Finset κ
    f : ι → κ → M
    ⊢ Eq (s.expect fun i => t.expect fun j => f i j) (t.expect fun j => s.expect f …
  -/
  rw [expect, expect, ← expect_sum_comm, ← expect_sum_comm, expect, expect, smul_comm, sum_comm]
  /-
    🎉 no goals
  -/


lemma expect_eq_zero (h : ∀ i ∈ s, f i = 0) : 𝔼 i ∈ s, f i = 0 :=
  (expect_congr rfl h).trans s.expect_const_zero


lemma exists_ne_zero_of_expect_ne_zero (h : 𝔼 i ∈ s, f i ≠ 0) : ∃ i ∈ s, f i ≠ 0 := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝¹ : AddCommMonoid M
    inst✝ : Module NNRat M
    s : Finset ι
    f : ι → M
    h : Ne (s.expect fun i => f i) 0
    ⊢ Exists fun i => And (Membership.mem s i) (Ne (f i) 0)
  -/
  contrapose! h; exact expect_eq_zero h
                 /-
                   🎉 no goals
                 -/


lemma expect_add_distrib (s : Finset ι) (f g : ι → M) :
    𝔼 i ∈ s, (f i + g i) = 𝔼 i ∈ s, f i + 𝔼 i ∈ s, g i := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝¹ : AddCommMonoid M
    inst✝ : Module NNRat M
    s : Finset ι
    f g : ι → M
    ⊢ Eq (s.expect fun i => HAdd.hAdd (f i) (g i)) (HAdd.hAdd (s.expect fun i => f …
  -/
  simp [expect, sum_add_distrib]
  /-
    🎉 no goals
  -/


lemma expect_add_expect_comm (f₁ f₂ g₁ g₂ : ι → M) :
    𝔼 i ∈ s, (f₁ i + f₂ i) + 𝔼 i ∈ s, (g₁ i + g₂ i) =
      𝔼 i ∈ s, (f₁ i + g₁ i) + 𝔼 i ∈ s, (f₂ i + g₂ i) := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝¹ : AddCommMonoid M
    inst✝ : Module NNRat M
    s : Finset ι
    f₁ f₂ g₁ g₂ : ι → M
    ⊢ Eq (HAdd.hAdd (s.expect fun i => HAdd.hAdd (f₁ i) (f₂ i)) (s.expect fun i => …
  -/
  simp_rw [expect_add_distrib, add_add_add_comm]
  /-
    🎉 no goals
  -/


lemma expect_eq_single_of_mem (i : ι) (hi : i ∈ s) (h : ∀ j ∈ s, j ≠ i → f j = 0) :
                                   /-
                                     ι : Type u_1
                                     M : Type u_3
                                     inst✝¹ : AddCommMonoid M
                                     inst✝ : Module NNRat M
                                     s : Finset ι
                                     f : ι → M
                                     i : ι
                                     hi : Membership.mem s i
                                     h : ∀ (j : ι), Membership.mem s j → Ne j i → Eq (f j) 0
                                     ⊢ Eq (s.expect fun i => f i) (HSMul.hSMul (Inv.inv ↑s.card) (f i))
                                   -/
    𝔼 i ∈ s, f i = f i /ℚ #s := by rw [expect, sum_eq_single_of_mem _ hi h]
                                   /-
                                     🎉 no goals
                                   -/


lemma expect_ite_zero (s : Finset ι) (p : ι → Prop) [DecidablePred p]
    (h : ∀ i ∈ s, ∀ j ∈ s, p i → p j → i = j) (a : M) :
    𝔼 i ∈ s, ite (p i) a 0 = ite (∃ i ∈ s, p i) (a /ℚ #s) 0 := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝² : AddCommMonoid M
    inst✝¹ : Module NNRat M
    s : Finset ι
    p : ι → Prop
    inst✝ : DecidablePred p
    h : ∀ (i : ι), Membership.mem s i → ∀ (j : ι), Membership.mem s j → p i → p j  …
    a : M
    ⊢ Eq (s.expect fun i => ite (p i) a 0) (ite (Exists fun i => And (Membership.m …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp [expect, sum_ite_zero _ _ h, *]
                /-
                  🎉 no goals
                -/


lemma expect_ite_mem (s t : Finset ι) (f : ι → M) :
    𝔼 i ∈ s, (if i ∈ t then f i else 0) = (#(s ∩ t) / #s : ℚ≥0) • 𝔼 i ∈ s ∩ t, f i := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝² : AddCommMonoid M
    inst✝¹ : Module NNRat M
    inst✝ : DecidableEq ι
    s t : Finset ι
    f : ι → M
    ⊢ Eq (s.expect fun i => ite (Membership.mem t i) (f i) 0) (HSMul.hSMul (HDiv.h …
  -/
  obtain hst | hst := (s ∩ t).eq_empty_or_nonempty
    /-
      case inl
      ι : Type u_1
      M : Type u_3
      inst✝² : AddCommMonoid M
      inst✝¹ : Module NNRat M
      inst✝ : DecidableEq ι
      s t : Finset ι
      f : ι → M
      hst : Eq (Inter.inter s t) EmptyCollection.emptyCollection
      ⊢ Eq (s.expect fun i => ite (Membership.mem t i) (f i) 0) (HSMul.hSMul (HDiv.h …
    -/
  · simp [expect, hst]
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Type u_1
      M : Type u_3
      inst✝² : AddCommMonoid M
      inst✝¹ : Module NNRat M
      inst✝ : DecidableEq ι
      s t : Finset ι
      f : ι → M
      hst : (Inter.inter s t).Nonempty
      ⊢ Eq (s.expect fun i => ite (Membership.mem t i) (f i) 0) (HSMul.hSMul (HDiv.h …
    -/
  · simp [expect, smul_smul, ← inv_mul_eq_div, hst.card_ne_zero]
    /-
      🎉 no goals
    -/


@[simp] lemma expect_dite_eq (i : ι) (f : ∀ j, i = j → M) :
    𝔼 j ∈ s, (if h : i = j then f j h else 0) = if i ∈ s then f i rfl /ℚ #s else 0 := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝² : AddCommMonoid M
    inst✝¹ : Module NNRat M
    s : Finset ι
    inst✝ : DecidableEq ι
    i : ι
    f : (j : ι) → Eq i j → M
    ⊢ Eq (s.expect fun j => dite (Eq i j) (fun h => f j h) fun h => 0) (ite (Membe …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp [expect, *]
                /-
                  🎉 no goals
                -/


@[simp] lemma expect_dite_eq' (i : ι) (f : ∀ j, j = i → M) :
    𝔼 j ∈ s, (if h : j = i then f j h else 0) = if i ∈ s then f i rfl /ℚ #s else 0 := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝² : AddCommMonoid M
    inst✝¹ : Module NNRat M
    s : Finset ι
    inst✝ : DecidableEq ι
    i : ι
    f : (j : ι) → Eq j i → M
    ⊢ Eq (s.expect fun j => dite (Eq j i) (fun h => f j h) fun h => 0) (ite (Membe …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp [expect, *]
                /-
                  🎉 no goals
                -/


@[simp] lemma expect_ite_eq (i : ι) (f : ι → M) :
    𝔼 j ∈ s, (if i = j then f j else 0) = if i ∈ s then f i /ℚ #s else 0 := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝² : AddCommMonoid M
    inst✝¹ : Module NNRat M
    s : Finset ι
    inst✝ : DecidableEq ι
    i : ι
    f : ι → M
    ⊢ Eq (s.expect fun j => ite (Eq i j) (f j) 0) (ite (Membership.mem s i) (HSMul …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp [expect, *]
                /-
                  🎉 no goals
                -/


@[simp] lemma expect_ite_eq' (i : ι) (f : ι → M) :
    𝔼 j ∈ s, (if j = i then f j else 0) = if i ∈ s then f i /ℚ #s else 0 := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝² : AddCommMonoid M
    inst✝¹ : Module NNRat M
    s : Finset ι
    inst✝ : DecidableEq ι
    i : ι
    f : ι → M
    ⊢ Eq (s.expect fun j => ite (Eq j i) (f j) 0) (ite (Membership.mem s i) (HSMul …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp [expect, *]
                /-
                  🎉 no goals
                -/


/-- Reorder an average.

The difference with `Finset.expect_bij'` is that the bijection is specified as a surjective
injection, rather than by an inverse function.

The difference with `Finset.expect_nbij` is that the bijection is allowed to use membership of the
domain of the average, rather than being a non-dependent function. -/
lemma expect_bij (i : ∀ a ∈ s, κ) (hi : ∀ a ha, i a ha ∈ t) (h : ∀ a ha, f a = g (i a ha))
    (i_inj : ∀ a₁ ha₁ a₂ ha₂, i a₁ ha₁ = i a₂ ha₂ → a₁ = a₂)
    (i_surj : ∀ b ∈ t, ∃ a ha, i a ha = b) : 𝔼 i ∈ s, f i = 𝔼 i ∈ t, g i := by
  /-
    ι : Type u_1
    κ : Type u_2
    M : Type u_3
    inst✝¹ : AddCommMonoid M
    inst✝ : Module NNRat M
    s : Finset ι
    f : ι → M
    t : Finset κ
    g : κ → M
    i : (a : ι) → Membership.mem s a → κ
    hi : ∀ (a : ι) (ha : Membership.mem s a), Membership.mem t (i a ha)
    h : ∀ (a : ι) (ha : Membership.mem s a), Eq (f a) (g (i a ha))
    i_inj : ∀ (a₁ : ι) (ha₁ : Membership.mem s a₁) (a₂ : ι) (ha₂ : Membership.mem  …
    i_surj : ∀ (b : κ), Membership.mem t b → Exists fun a => Exists fun ha => Eq ( …
    ⊢ Eq (s.expect fun i => f i) (t.expect fun i => g i)
  -/
  simp_rw [expect, card_bij i hi i_inj i_surj, sum_bij i hi i_inj i_surj h]
  /-
    🎉 no goals
  -/


/-- Reorder an average.

The difference with `Finset.expect_bij` is that the bijection is specified with an inverse, rather
than as a surjective injection.

The difference with `Finset.expect_nbij'` is that the bijection and its inverse are allowed to use
membership of the domains of the averages, rather than being non-dependent functions. -/
lemma expect_bij' (i : ∀ a ∈ s, κ) (j : ∀ a ∈ t, ι) (hi : ∀ a ha, i a ha ∈ t)
    (hj : ∀ a ha, j a ha ∈ s) (left_inv : ∀ a ha, j (i a ha) (hi a ha) = a)
    (right_inv : ∀ a ha, i (j a ha) (hj a ha) = a) (h : ∀ a ha, f a = g (i a ha)) :
    𝔼 i ∈ s, f i = 𝔼 i ∈ t, g i := by
  /-
    ι : Type u_1
    κ : Type u_2
    M : Type u_3
    inst✝¹ : AddCommMonoid M
    inst✝ : Module NNRat M
    s : Finset ι
    f : ι → M
    t : Finset κ
    g : κ → M
    i : (a : ι) → Membership.mem s a → κ
    j : (a : κ) → Membership.mem t a → ι
    hi : ∀ (a : ι) (ha : Membership.mem s a), Membership.mem t (i a ha)
    hj : ∀ (a : κ) (ha : Membership.mem t a), Membership.mem s (j a ha)
    left_inv : ∀ (a : ι) (ha : Membership.mem s a), Eq (j (i a ha) ⋯) a
    right_inv : ∀ (a : κ) (ha : Membership.mem t a), Eq (i (j a ha) ⋯) a
    h : ∀ (a : ι) (ha : Membership.mem s a), Eq (f a) (g (i a ha))
    ⊢ Eq (s.expect fun i => f i) (t.expect fun i => g i)
  -/
  simp_rw [expect, card_bij' i j hi hj left_inv right_inv, sum_bij' i j hi hj left_inv right_inv h]
  /-
    🎉 no goals
  -/


/-- Reorder an average.

The difference with `Finset.expect_nbij'` is that the bijection is specified as a surjective
injection, rather than by an inverse function.

The difference with `Finset.expect_bij` is that the bijection is a non-dependent function, rather
than being allowed to use membership of the domain of the average. -/
lemma expect_nbij (i : ι → κ) (hi : ∀ a ∈ s, i a ∈ t) (h : ∀ a ∈ s, f a = g (i a))
    (i_inj : (s : Set ι).InjOn i) (i_surj : (s : Set ι).SurjOn i t) :
    𝔼 i ∈ s, f i = 𝔼 i ∈ t, g i := by
  /-
    ι : Type u_1
    κ : Type u_2
    M : Type u_3
    inst✝¹ : AddCommMonoid M
    inst✝ : Module NNRat M
    s : Finset ι
    f : ι → M
    t : Finset κ
    g : κ → M
    i : ι → κ
    hi : ∀ (a : ι), Membership.mem s a → Membership.mem t (i a)
    h : ∀ (a : ι), Membership.mem s a → Eq (f a) (g (i a))
    i_inj : Set.InjOn i ↑s
    i_surj : Set.SurjOn i ↑s ↑t
    ⊢ Eq (s.expect fun i => f i) (t.expect fun i => g i)
  -/
  simp_rw [expect, card_nbij i hi i_inj i_surj, sum_nbij i hi i_inj i_surj h]
  /-
    🎉 no goals
  -/


/-- Reorder an average.

The difference with `Finset.expect_nbij` is that the bijection is specified with an inverse, rather
than as a surjective injection.

The difference with `Finset.expect_bij'` is that the bijection and its inverse are non-dependent
functions, rather than being allowed to use membership of the domains of the averages.

The difference with `Finset.expect_equiv` is that bijectivity is only required to hold on the
domains of the averages, rather than on the entire types. -/
lemma expect_nbij' (i : ι → κ) (j : κ → ι) (hi : ∀ a ∈ s, i a ∈ t) (hj : ∀ a ∈ t, j a ∈ s)
    (left_inv : ∀ a ∈ s, j (i a) = a) (right_inv : ∀ a ∈ t, i (j a) = a)
    (h : ∀ a ∈ s, f a = g (i a)) : 𝔼 i ∈ s, f i = 𝔼 i ∈ t, g i := by
  simp_rw [expect, card_nbij' i j hi hj left_inv right_inv,
    sum_nbij' i j hi hj left_inv right_inv h]


/-- `Finset.expect_equiv` is a specialization of `Finset.expect_bij` that automatically fills in
most arguments. -/
lemma expect_equiv (e : ι ≃ κ) (hst : ∀ i, i ∈ s ↔ e i ∈ t) (hfg : ∀ i ∈ s, f i = g (e i)) :
                                      /-
                                        ι : Type u_1
                                        κ : Type u_2
                                        M : Type u_3
                                        inst✝¹ : AddCommMonoid M
                                        inst✝ : Module NNRat M
                                        s : Finset ι
                                        f : ι → M
                                        t : Finset κ
                                        g : κ → M
                                        e : Equiv ι κ
                                        hst : ∀ (i : ι), Iff (Membership.mem s i) (Membership.mem t (e i))
                                        hfg : ∀ (i : ι), Membership.mem s i → Eq (f i) (g (e i))
                                        ⊢ Eq (s.expect fun i => f i) (t.expect fun i => g i)
                                      -/
    𝔼 i ∈ s, f i = 𝔼 i ∈ t, g i := by simp_rw [expect, card_equiv e hst, sum_equiv e hst hfg]
                                      /-
                                        🎉 no goals
                                      -/


/-- Expectation over a product set equals the expectation of the fiberwise expectations.

For rewriting in the reverse direction, use `Finset.expect_product'`. -/
lemma expect_product (s : Finset ι) (t : Finset κ) (f : ι × κ → M) :
    𝔼 x ∈ s ×ˢ t, f x = 𝔼 i ∈ s, 𝔼 j ∈ t, f (i, j) := by
  /-
    ι : Type u_1
    κ : Type u_2
    M : Type u_3
    inst✝¹ : AddCommMonoid M
    inst✝ : Module NNRat M
    s : Finset ι
    t : Finset κ
    f : Prod ι κ → M
    ⊢ Eq ((SProd.sprod s t).expect fun x => f x) (s.expect fun i => t.expect fun j …
  -/
  simp only [expect, card_product, sum_product, smul_sum, mul_inv, mul_smul, Nat.cast_mul]
  /-
    🎉 no goals
  -/


/-- Expectation over a product set equals the expectation of the fiberwise expectations.

For rewriting in the reverse direction, use `Finset.expect_product`. -/
lemma expect_product' (s : Finset ι) (t : Finset κ) (f : ι → κ → M) :
    𝔼 i ∈ s ×ˢ t, f i.1 i.2 = 𝔼 i ∈ s, 𝔼 j ∈ t, f i j := by
  /-
    ι : Type u_1
    κ : Type u_2
    M : Type u_3
    inst✝¹ : AddCommMonoid M
    inst✝ : Module NNRat M
    s : Finset ι
    t : Finset κ
    f : ι → κ → M
    ⊢ Eq ((SProd.sprod s t).expect fun i => f i.1 i.2) (s.expect fun i => t.expect …
  -/
  simp only [expect, card_product, sum_product', smul_sum, mul_inv, mul_smul, Nat.cast_mul]
  /-
    🎉 no goals
  -/


@[simp]
lemma expect_image [DecidableEq ι] {m : κ → ι} (hm : (t : Set κ).InjOn m) :
    𝔼 i ∈ t.image m, f i = 𝔼 i ∈ t, f (m i) := by
  /-
    ι : Type u_1
    κ : Type u_2
    M : Type u_3
    inst✝² : AddCommMonoid M
    inst✝¹ : Module NNRat M
    f : ι → M
    t : Finset κ
    inst✝ : DecidableEq ι
    m : κ → ι
    hm : Set.InjOn m ↑t
    ⊢ Eq ((Finset.image m t).expect fun i => f i) (t.expect fun i => f (m i))
  -/
  simp_rw [expect, card_image_of_injOn hm, sum_image hm]
  /-
    🎉 no goals
  -/


@[simp] lemma expect_inv_index [DecidableEq ι] [InvolutiveInv ι] (s : Finset ι) (f : ι → M) :
    𝔼 i ∈ s⁻¹, f i = 𝔼 i ∈ s, f i⁻¹ := expect_image inv_injective.injOn


@[simp] lemma expect_neg_index [DecidableEq ι] [InvolutiveNeg ι] (s : Finset ι) (f : ι → M) :
    𝔼 i ∈ -s, f i = 𝔼 i ∈ s, f (-i) := expect_image neg_injective.injOn


lemma _root_.map_expect {F : Type*} [FunLike F M N] [LinearMapClass F ℚ≥0 M N]
    (g : F) (f : ι → M) (s : Finset ι) :
                                              /-
                                                ι : Type u_1
                                                M : Type u_3
                                                N : Type u_4
                                                inst✝⁵ : AddCommMonoid M
                                                inst✝⁴ : Module NNRat M
                                                inst✝³ : AddCommMonoid N
                                                inst✝² : Module NNRat N
                                                F : Type u_5
                                                inst✝¹ : FunLike F M N
                                                inst✝ : LinearMapClass F NNRat M N
                                                g : F
                                                f : ι → M
                                                s : Finset ι
                                                ⊢ Eq (g (s.expect fun i => f i)) (s.expect fun i => g (f i))
                                              -/
    g (𝔼 i ∈ s, f i) = 𝔼 i ∈ s, g (f i) := by simp only [expect, map_smul, map_natCast, map_sum]
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
lemma card_smul_expect (s : Finset ι) (f : ι → M) : #s • 𝔼 i ∈ s, f i = ∑ i ∈ s, f i := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝¹ : AddCommMonoid M
    inst✝ : Module NNRat M
    s : Finset ι
    f : ι → M
    ⊢ Eq (HSMul.hSMul s.card (s.expect fun i => f i)) (s.sum fun i => f i)
  -/
  obtain rfl | hs := s.eq_empty_or_nonempty
    /-
      case inl
      ι : Type u_1
      M : Type u_3
      inst✝¹ : AddCommMonoid M
      inst✝ : Module NNRat M
      f : ι → M
      ⊢ Eq (HSMul.hSMul EmptyCollection.emptyCollection.card (EmptyCollection.emptyC …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Type u_1
      M : Type u_3
      inst✝¹ : AddCommMonoid M
      inst✝ : Module NNRat M
      s : Finset ι
      f : ι → M
      hs : s.Nonempty
      ⊢ Eq (HSMul.hSMul s.card (s.expect fun i => f i)) (s.sum fun i => f i)
    -/
  · rw [expect, ← Nat.cast_smul_eq_nsmul ℚ≥0, smul_inv_smul₀]
    /-
      case inr.ha
      ι : Type u_1
      M : Type u_3
      inst✝¹ : AddCommMonoid M
      inst✝ : Module NNRat M
      s : Finset ι
      f : ι → M
      hs : s.Nonempty
      ⊢ Ne (↑s.card) 0
    -/
    exact mod_cast hs.card_ne_zero
    /-
      🎉 no goals
    -/


@[simp] lemma _root_.Fintype.card_smul_expect [Fintype ι] (f : ι → M) :
    Fintype.card ι • 𝔼 i, f i = ∑ i, f i := Finset.card_smul_expect _ _


@[simp] lemma expect_const (hs : s.Nonempty) (a : M) : 𝔼 _i ∈ s, a = a := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝¹ : AddCommMonoid M
    inst✝ : Module NNRat M
    s : Finset ι
    hs : s.Nonempty
    a : M
    ⊢ Eq (s.expect fun _i => a) a
  -/
  rw [expect, sum_const, ← Nat.cast_smul_eq_nsmul ℚ≥0, inv_smul_smul₀]
  /-
    case ha
    ι : Type u_1
    M : Type u_3
    inst✝¹ : AddCommMonoid M
    inst✝ : Module NNRat M
    s : Finset ι
    hs : s.Nonempty
    a : M
    ⊢ Ne (↑s.card) 0
  -/
  exact mod_cast hs.card_ne_zero
  /-
    🎉 no goals
  -/


lemma smul_expect {G : Type*} [DistribSMul G M] [SMulCommClass G ℚ≥0 M] (a : G)
    (s : Finset ι) (f : ι → M) : a • 𝔼 i ∈ s, f i = 𝔼 i ∈ s, a • f i := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝³ : AddCommMonoid M
    inst✝² : Module NNRat M
    G : Type u_5
    inst✝¹ : DistribSMul G M
    inst✝ : SMulCommClass G NNRat M
    a : G
    s : Finset ι
    f : ι → M
    ⊢ Eq (HSMul.hSMul a (s.expect fun i => f i)) (s.expect fun i => HSMul.hSMul a  …
  -/
  simp only [expect, smul_sum, smul_comm]
  /-
    🎉 no goals
  -/


lemma expect_sub_distrib (s : Finset ι) (f g : ι → M) :
    𝔼 i ∈ s, (f i - g i) = 𝔼 i ∈ s, f i - 𝔼 i ∈ s, g i := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝¹ : AddCommGroup M
    inst✝ : Module NNRat M
    s : Finset ι
    f g : ι → M
    ⊢ Eq (s.expect fun i => HSub.hSub (f i) (g i)) (HSub.hSub (s.expect fun i => f …
  -/
  simp only [expect, sum_sub_distrib, smul_sub]
  /-
    🎉 no goals
  -/


@[simp]
lemma expect_neg_distrib (s : Finset ι) (f : ι → M) : 𝔼 i ∈ s, -f i = -𝔼 i ∈ s, f i := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝¹ : AddCommGroup M
    inst✝ : Module NNRat M
    s : Finset ι
    f : ι → M
    ⊢ Eq (s.expect fun i => Neg.neg (f i)) (Neg.neg (s.expect fun i => f i))
  -/
  simp [expect]
  /-
    🎉 no goals
  -/


@[simp] lemma card_mul_expect (s : Finset ι) (f : ι → M) :
                                           /-
                                             ι : Type u_1
                                             M : Type u_3
                                             inst✝¹ : Semiring M
                                             inst✝ : Module NNRat M
                                             s : Finset ι
                                             f : ι → M
                                             ⊢ Eq (HMul.hMul (↑s.card) (s.expect fun i => f i)) (s.sum fun i => f i)
                                           -/
    #s * 𝔼 i ∈ s, f i = ∑ i ∈ s, f i := by rw [← nsmul_eq_mul, card_smul_expect]
                                           /-
                                             🎉 no goals
                                           -/


@[simp] lemma _root_.Fintype.card_mul_expect [Fintype ι] (f : ι → M) :
    Fintype.card ι * 𝔼 i, f i = ∑ i, f i := Finset.card_mul_expect _ _


lemma expect_mul [IsScalarTower ℚ≥0 M M] (s : Finset ι) (f : ι → M) (a : M) :
                                                /-
                                                  ι : Type u_1
                                                  M : Type u_3
                                                  inst✝² : Semiring M
                                                  inst✝¹ : Module NNRat M
                                                  inst✝ : IsScalarTower NNRat M M
                                                  s : Finset ι
                                                  f : ι → M
                                                  a : M
                                                  ⊢ Eq (HMul.hMul (s.expect fun i => f i) a) (s.expect fun i => HMul.hMul (f i) a)
                                                -/
    (𝔼 i ∈ s, f i) * a = 𝔼 i ∈ s, f i * a := by rw [expect, expect, smul_mul_assoc, sum_mul]
                                                /-
                                                  🎉 no goals
                                                -/


lemma mul_expect [SMulCommClass ℚ≥0 M M] (s : Finset ι) (f : ι → M) (a : M) :
                                              /-
                                                ι : Type u_1
                                                M : Type u_3
                                                inst✝² : Semiring M
                                                inst✝¹ : Module NNRat M
                                                inst✝ : SMulCommClass NNRat M M
                                                s : Finset ι
                                                f : ι → M
                                                a : M
                                                ⊢ Eq (HMul.hMul a (s.expect fun i => f i)) (s.expect fun i => HMul.hMul a (f i))
                                              -/
    a * 𝔼 i ∈ s, f i = 𝔼 i ∈ s, a * f i := by rw [expect, expect, mul_smul_comm, mul_sum]
                                              /-
                                                🎉 no goals
                                              -/


lemma expect_mul_expect [IsScalarTower ℚ≥0 M M] [SMulCommClass ℚ≥0 M M] (s : Finset ι)
    (t : Finset κ) (f : ι → M) (g : κ → M) :
    (𝔼 i ∈ s, f i) * 𝔼 j ∈ t, g j = 𝔼 i ∈ s, 𝔼 j ∈ t, f i * g j := by
  /-
    ι : Type u_1
    κ : Type u_2
    M : Type u_3
    inst✝³ : Semiring M
    inst✝² : Module NNRat M
    inst✝¹ : IsScalarTower NNRat M M
    inst✝ : SMulCommClass NNRat M M
    s : Finset ι
    t : Finset κ
    f : ι → M
    g : κ → M
    ⊢ Eq (HMul.hMul (s.expect fun i => f i) (t.expect fun j => g j)) (s.expect fun …
  -/
  simp_rw [expect_mul, mul_expect]
  /-
    🎉 no goals
  -/


lemma expect_pow (s : Finset ι) (f : ι → M) (n : ℕ) :
    (𝔼 i ∈ s, f i) ^ n = 𝔼 p ∈ Fintype.piFinset fun _ : Fin n ↦ s, ∏ i, f (p i) := by
  classical
  rw [expect, smul_pow, sum_pow', expect, Fintype.card_piFinset_const, inv_pow, Nat.cast_pow]


lemma expect_boole_mul [Fintype ι] [Nonempty ι] [DecidableEq ι] (f : ι → M) (i : ι) :
    𝔼 j, ite (i = j) (Fintype.card ι : M) 0 * f j = f i := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝⁴ : Semifield M
    inst✝³ : CharZero M
    inst✝² : Fintype ι
    inst✝¹ : Nonempty ι
    inst✝ : DecidableEq ι
    f : ι → M
    i : ι
    ⊢ Eq (Finset.univ.expect fun j => HMul.hMul (ite (Eq i j) (↑(Fintype.card ι))  …
  -/
  simp_rw [expect_univ, ite_mul, zero_mul, sum_ite_eq, if_pos (mem_univ _)]
  /-
    ι : Type u_1
    M : Type u_3
    inst✝⁴ : Semifield M
    inst✝³ : CharZero M
    inst✝² : Fintype ι
    inst✝¹ : Nonempty ι
    inst✝ : DecidableEq ι
    f : ι → M
    i : ι
    ⊢ Eq (HSMul.hSMul (Inv.inv ↑(Fintype.card ι)) (HMul.hMul (↑(Fintype.card ι)) ( …
  -/
  rw [← @NNRat.cast_natCast M, ← NNRat.smul_def, inv_smul_smul₀]
  /-
    case ha
    ι : Type u_1
    M : Type u_3
    inst✝⁴ : Semifield M
    inst✝³ : CharZero M
    inst✝² : Fintype ι
    inst✝¹ : Nonempty ι
    inst✝ : DecidableEq ι
    f : ι → M
    i : ι
    ⊢ Ne (↑(Fintype.card ι)) 0
  -/
  simp [Fintype.card_ne_zero]
  /-
    🎉 no goals
  -/


lemma expect_boole_mul' [Fintype ι] [Nonempty ι] [DecidableEq ι] (f : ι → M) (i : ι) :
    𝔼 j, ite (j = i) (Fintype.card ι : M) 0 * f j = f i := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝⁴ : Semifield M
    inst✝³ : CharZero M
    inst✝² : Fintype ι
    inst✝¹ : Nonempty ι
    inst✝ : DecidableEq ι
    f : ι → M
    i : ι
    ⊢ Eq (Finset.univ.expect fun j => HMul.hMul (ite (Eq j i) (↑(Fintype.card ι))  …
  -/
  simp_rw [@eq_comm _ _ i, expect_boole_mul]
  /-
    🎉 no goals
  -/


lemma expect_eq_sum_div_card (s : Finset ι) (f : ι → M) :
    𝔼 i ∈ s, f i = (∑ i ∈ s, f i) / #s := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝¹ : Semifield M
    inst✝ : CharZero M
    s : Finset ι
    f : ι → M
    ⊢ Eq (s.expect fun i => f i) (HDiv.hDiv (s.sum fun i => f i) ↑s.card)
  -/
  rw [expect, NNRat.smul_def, div_eq_inv_mul, NNRat.cast_inv, NNRat.cast_natCast]
  /-
    🎉 no goals
  -/


lemma _root_.Fintype.expect_eq_sum_div_card [Fintype ι] (f : ι → M) :
    𝔼 i, f i = (∑ i, f i) / Fintype.card ι := Finset.expect_eq_sum_div_card _ _


lemma expect_div (s : Finset ι) (f : ι → M) (a : M) : (𝔼 i ∈ s, f i) / a = 𝔼 i ∈ s, f i / a := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝¹ : Semifield M
    inst✝ : CharZero M
    s : Finset ι
    f : ι → M
    a : M
    ⊢ Eq (HDiv.hDiv (s.expect fun i => f i) a) (s.expect fun i => HDiv.hDiv (f i) a)
  -/
  simp_rw [div_eq_mul_inv, expect_mul]
  /-
    🎉 no goals
  -/


@[simp] lemma expect_apply {α : Type*} {π : α → Type*} [∀ a, CommSemiring (π a)]
    [∀ a, Module ℚ≥0 (π a)] (s : Finset ι) (f : ι → ∀ a, π a) (a : α) :
                                            /-
                                              ι : Type u_1
                                              α : Type u_5
                                              π : α → Type u_6
                                              inst✝¹ : (a : α) → CommSemiring (π a)
                                              inst✝ : (a : α) → Module NNRat (π a)
                                              s : Finset ι
                                              f : ι → (a : α) → π a
                                              a : α
                                              ⊢ Eq (s.expect (fun i => f i) a) (s.expect fun i => f i a)
                                            -/
    (𝔼 i ∈ s, f i) a = 𝔼 i ∈ s, f i a := by simp [expect]
                                            /-
                                              🎉 no goals
                                            -/


@[simp, norm_cast]
lemma coe_expect (s : Finset ι) (f : ι → M) : 𝔼 i ∈ s, f i = 𝔼 i ∈ s, (f i : N) :=
  map_expect (algebraMap _ _) _ _


/-- `Fintype.expect_bijective` is a variant of `Finset.expect_bij` that accepts
`Function.Bijective`.

See `Function.Bijective.expect_comp` for a version without `h`. -/
lemma expect_bijective (e : ι → κ) (he : Bijective e) (f : ι → M) (g : κ → M)
    (h : ∀ i, f i = g (e i)) : 𝔼 i, f i = 𝔼 i, g i :=
  expect_nbij e (fun _ _ ↦ mem_univ _) (fun i _ ↦ h i) he.injective.injOn <| by
    /-
      ι : Type u_1
      κ : Type u_2
      M : Type u_3
      inst✝³ : Fintype ι
      inst✝² : Fintype κ
      inst✝¹ : AddCommMonoid M
      inst✝ : Module NNRat M
      e : ι → κ
      he : Function.Bijective e
      f : ι → M
      g : κ → M
      h : ∀ (i : ι), Eq (f i) (g (e i))
      ⊢ Set.SurjOn e ↑Finset.univ ↑Finset.univ
    -/
    simpa using he.surjective.surjOn _
    /-
      🎉 no goals
    -/


/-- `Fintype.expect_equiv` is a specialization of `Finset.expect_bij` that automatically fills in
most arguments.

See `Equiv.expect_comp` for a version without `h`. -/
lemma expect_equiv (e : ι ≃ κ) (f : ι → M) (g : κ → M) (h : ∀ i, f i = g (e i)) :
    𝔼 i, f i = 𝔼 i, g i := expect_bijective _ e.bijective f g h


lemma expect_const [Nonempty ι] (a : M) : 𝔼 _i : ι, a = a := Finset.expect_const univ_nonempty _


lemma expect_ite_zero (p : ι → Prop) [DecidablePred p] (h : ∀ i j, p i → p j → i = j) (a : M) :
    𝔼 i, ite (p i) a 0 = ite (∃ i, p i) (a /ℚ Fintype.card ι) 0 := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝³ : Fintype ι
    inst✝² : AddCommMonoid M
    inst✝¹ : Module NNRat M
    p : ι → Prop
    inst✝ : DecidablePred p
    h : ∀ (i j : ι), p i → p j → Eq i j
    a : M
    ⊢ Eq (Finset.univ.expect fun i => ite (p i) a 0) (ite (Exists fun i => p i) (H …
  -/
  simp [univ.expect_ite_zero p (by simpa using h), card_univ]
  /-
    🎉 no goals
  -/


@[simp] lemma expect_ite_mem (s : Finset ι) (f : ι → M) :
    𝔼 i, (if i ∈ s then f i else 0) = s.dens • 𝔼 i ∈ s, f i := by
  /-
    ι : Type u_1
    M : Type u_3
    inst✝³ : Fintype ι
    inst✝² : AddCommMonoid M
    inst✝¹ : Module NNRat M
    inst✝ : DecidableEq ι
    s : Finset ι
    f : ι → M
    ⊢ Eq (Finset.univ.expect fun i => ite (Membership.mem s i) (f i) 0) (HSMul.hSM …
  -/
  simp [Finset.expect_ite_mem, dens]
  /-
    🎉 no goals
  -/


lemma expect_dite_eq (i : ι) (f : ∀ j, i = j → M) :
                                                                    /-
                                                                      ι : Type u_1
                                                                      M : Type u_3
                                                                      inst✝³ : Fintype ι
                                                                      inst✝² : AddCommMonoid M
                                                                      inst✝¹ : Module NNRat M
                                                                      inst✝ : DecidableEq ι
                                                                      i : ι
                                                                      f : (j : ι) → Eq i j → M
                                                                      ⊢ Eq (Finset.univ.expect fun j => dite (Eq i j) (fun h => f j h) fun h => 0) ( …
                                                                    -/
    𝔼 j, (if h : i = j then f j h else 0) = f i rfl /ℚ card ι := by simp [card_univ]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


lemma expect_dite_eq' (i : ι) (f : ∀ j, j = i → M) :
                                                                    /-
                                                                      ι : Type u_1
                                                                      M : Type u_3
                                                                      inst✝³ : Fintype ι
                                                                      inst✝² : AddCommMonoid M
                                                                      inst✝¹ : Module NNRat M
                                                                      inst✝ : DecidableEq ι
                                                                      i : ι
                                                                      f : (j : ι) → Eq j i → M
                                                                      ⊢ Eq (Finset.univ.expect fun j => dite (Eq j i) (fun h => f j h) fun h => 0) ( …
                                                                    -/
    𝔼 j, (if h : j = i then f j h else 0) = f i rfl /ℚ card ι := by simp [card_univ]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


lemma expect_ite_eq (i : ι) (f : ι → M) :
                                                          /-
                                                            ι : Type u_1
                                                            M : Type u_3
                                                            inst✝³ : Fintype ι
                                                            inst✝² : AddCommMonoid M
                                                            inst✝¹ : Module NNRat M
                                                            inst✝ : DecidableEq ι
                                                            i : ι
                                                            f : ι → M
                                                            ⊢ Eq (Finset.univ.expect fun j => ite (Eq i j) (f j) 0) (HSMul.hSMul (Inv.inv  …
                                                          -/
    𝔼 j, (if i = j then f j else 0) = f i /ℚ card ι := by simp [card_univ]
                                                          /-
                                                            🎉 no goals
                                                          -/


lemma expect_ite_eq' (i : ι) (f : ι → M) :
                                                          /-
                                                            ι : Type u_1
                                                            M : Type u_3
                                                            inst✝³ : Fintype ι
                                                            inst✝² : AddCommMonoid M
                                                            inst✝¹ : Module NNRat M
                                                            inst✝ : DecidableEq ι
                                                            i : ι
                                                            f : ι → M
                                                            ⊢ Eq (Finset.univ.expect fun j => ite (Eq j i) (f j) 0) (HSMul.hSMul (Inv.inv  …
                                                          -/
    𝔼 j, (if j = i then f j else 0) = f i /ℚ card ι := by simp [card_univ]
                                                          /-
                                                            🎉 no goals
                                                          -/


lemma expect_one [Nonempty ι] : 𝔼 _i : ι, (1 : M) = 1 := expect_const _


lemma expect_mul_expect [IsScalarTower ℚ≥0 M M] [SMulCommClass ℚ≥0 M M] (f : ι → M)
    (g : κ → M) : (𝔼 i, f i) * 𝔼 j, g j = 𝔼 i, 𝔼 j, f i * g j :=
  Finset.expect_mul_expect ..


