local notation "q" => Fintype.card K


/-- The cardinality of a field is at most `n` times the cardinality of the image of a degree `n`
  polynomial -/
theorem card_image_polynomial_eval [DecidableEq R] [Fintype R] {p : R[X]} (hp : 0 < p.degree) :
    Fintype.card R ≤ natDegree p * #(univ.image fun x => eval x p) :=
  Finset.card_le_mul_card_image _ _ (fun a _ =>
    calc
      _ = #(p - C a).roots.toFinset :=
                           /-
                             R : Type u_2
                             inst✝³ : CommRing R
                             inst✝² : IsDomain R
                             inst✝¹ : DecidableEq R
                             inst✝ : Fintype R
                             p : Polynomial R
                             hp : LT.lt 0 p.degree
                             a : R
                             x✝ : Membership.mem (Finset.image (fun x => Polynomial.eval x p) Finset.univ) a
                             ⊢ Eq (Finset.filter (fun a_1 => Eq (Polynomial.eval a_1 p) a) Finset.univ) (HS …
                           -/
        congr_arg card (by simp [Finset.ext_iff, ← mem_roots_sub_C hp])
                           /-
                             🎉 no goals
                           -/
      _ ≤ Multiset.card (p - C a).roots := Multiset.toFinset_card_le _
      _ ≤ _ := card_roots_sub_C' hp)


/-- If `f` and `g` are quadratic polynomials, then the `f.eval a + g.eval b = 0` has a solution. -/
theorem exists_root_sum_quadratic [Fintype R] {f g : R[X]} (hf2 : degree f = 2) (hg2 : degree g = 2)
    (hR : Fintype.card R % 2 = 1) : ∃ a b, f.eval a + g.eval b = 0 :=
  letI := Classical.decEq R
  suffices ¬Disjoint (univ.image fun x : R => eval x f)
    (univ.image fun x : R => eval x (-g)) by
    /-
      R : Type u_2
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : Fintype R
      f g : Polynomial R
      hf2 : Eq f.degree 2
      hg2 : Eq g.degree 2
      hR : Eq (HMod.hMod (Fintype.card R) 2) 1
      this✝ : DecidableEq R := Classical.decEq R
      this : Not (Disjoint (Finset.image (fun x => Polynomial.eval x f) Finset.univ) …
      ⊢ Exists fun a => Exists fun b => Eq (HAdd.hAdd (Polynomial.eval a f) (Polynom …
    -/
    simp only [disjoint_left, mem_image] at this
    /-
      R : Type u_2
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : Fintype R
      f g : Polynomial R
      hf2 : Eq f.degree 2
      hg2 : Eq g.degree 2
      hR : Eq (HMod.hMod (Fintype.card R) 2) 1
      this✝ : DecidableEq R := Classical.decEq R
      this : Not (∀ ⦃a : R⦄, (Exists fun a_1 => And (Membership.mem Finset.univ a_1) …
      ⊢ Exists fun a => Exists fun b => Eq (HAdd.hAdd (Polynomial.eval a f) (Polynom …
    -/
    push_neg at this
    /-
      R : Type u_2
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : Fintype R
      f g : Polynomial R
      hf2 : Eq f.degree 2
      hg2 : Eq g.degree 2
      hR : Eq (HMod.hMod (Fintype.card R) 2) 1
      this✝ : DecidableEq R := Classical.decEq R
      this : Exists fun ⦃a⦄ => And (Exists fun a_1 => And (Membership.mem Finset.uni …
      ⊢ Exists fun a => Exists fun b => Eq (HAdd.hAdd (Polynomial.eval a f) (Polynom …
    -/
    rcases this with ⟨x, ⟨a, _, ha⟩, ⟨b, _, hb⟩⟩
    /-
      case intro.intro.intro.intro.intro.intro
      R : Type u_2
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : Fintype R
      f g : Polynomial R
      hf2 : Eq f.degree 2
      hg2 : Eq g.degree 2
      hR : Eq (HMod.hMod (Fintype.card R) 2) 1
      this : DecidableEq R := Classical.decEq R
      x a : R
      left✝¹ : Membership.mem Finset.univ a
      ha : Eq (Polynomial.eval a f) x
      b : R
      left✝ : Membership.mem Finset.univ b
      hb : Eq (Polynomial.eval b (Neg.neg g)) x
      ⊢ Exists fun a => Exists fun b => Eq (HAdd.hAdd (Polynomial.eval a f) (Polynom …
    -/
    exact ⟨a, b, by rw [ha, ← hb, eval_neg, neg_add_cancel]⟩
    /-
      🎉 no goals
    -/
  fun hd : Disjoint _ _ =>
  lt_irrefl (2 * #((univ.image fun x : R => eval x f) ∪ univ.image fun x : R => eval x (-g))) <|
    calc 2 * #((univ.image fun x : R => eval x f) ∪ univ.image fun x : R => eval x (-g))
                                                          /-
                                                            R : Type u_2
                                                            inst✝² : CommRing R
                                                            inst✝¹ : IsDomain R
                                                            inst✝ : Fintype R
                                                            f g : Polynomial R
                                                            hf2 : Eq f.degree 2
                                                            hg2 : Eq g.degree 2
                                                            hR : Eq (HMod.hMod (Fintype.card R) 2) 1
                                                            this : DecidableEq R := Classical.decEq R
                                                            hd : Disjoint (Finset.image (fun x => Polynomial.eval x f) Finset.univ) (Finse …
                                                            ⊢ LT.lt 0 f.degree
                                                          -/
        ≤ 2 * Fintype.card R := Nat.mul_le_mul_left _ (Finset.card_le_univ _)
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
                                        /-
                                          R : Type u_2
                                          inst✝² : CommRing R
                                          inst✝¹ : IsDomain R
                                          inst✝ : Fintype R
                                          f g : Polynomial R
                                          hf2 : Eq f.degree 2
                                          hg2 : Eq g.degree 2
                                          hR : Eq (HMod.hMod (Fintype.card R) 2) 1
                                          this : DecidableEq R := Classical.decEq R
                                          hd : Disjoint (Finset.image (fun x => Polynomial.eval x f) Finset.univ) (Finse …
                                          ⊢ Not (Eq (HMod.hMod (Fintype.card R) 2) (HMod.hMod (HMul.hMul f.natDegree (Fi …
                                        -/
      _ = Fintype.card R + Fintype.card R := two_mul _
                                        /-
                                          🎉 no goals
                                        -/
                                          /-
                                            R : Type u_2
                                            inst✝² : CommRing R
                                            inst✝¹ : IsDomain R
                                            inst✝ : Fintype R
                                            f g : Polynomial R
                                            hf2 : Eq f.degree 2
                                            hg2 : Eq g.degree 2
                                            hR : Eq (HMod.hMod (Fintype.card R) 2) 1
                                            this : DecidableEq R := Classical.decEq R
                                            hd : Disjoint (Finset.image (fun x => Polynomial.eval x f) Finset.univ) (Finse …
                                            ⊢ LT.lt 0 (Neg.neg g).degree
                                          -/
      _ < natDegree f * #(univ.image fun x : R => eval x f) +
                                                                /-
                                                                  🎉 no goals
                                                                -/
            natDegree (-g) * #(univ.image fun x : R => eval x (-g)) :=
        /-
          R : Type u_2
          inst✝² : CommRing R
          inst✝¹ : IsDomain R
          inst✝ : Fintype R
          f g : Polynomial R
          hf2 : Eq f.degree 2
          hg2 : Eq g.degree 2
          hR : Eq (HMod.hMod (Fintype.card R) 2) 1
          this : DecidableEq R := Classical.decEq R
          hd : Disjoint (Finset.image (fun x => Polynomial.eval x f) Finset.univ) (Finse …
          ⊢ Eq (HAdd.hAdd (HMul.hMul f.natDegree (Finset.image (fun x => Polynomial.eval …
        -/
        (add_lt_add_of_lt_of_le
        /-
          R : Type u_2
          inst✝² : CommRing R
          inst✝¹ : IsDomain R
          inst✝ : Fintype R
          f g : Polynomial R
          hf2 : Eq f.degree 2
          hg2 : Eq g.degree 2
          hR : Eq (HMod.hMod (Fintype.card R) 2) 1
          this : DecidableEq R := Classical.decEq R
          hd : Disjoint (Finset.image (fun x => Polynomial.eval x f) Finset.univ) (Finse …
          ⊢ Eq (HAdd.hAdd (HMul.hMul f.natDegree (Finset.image (fun x => Polynomial.eval …
        -/
          (lt_of_le_of_ne (card_image_polynomial_eval (by rw [hf2]; decide))
        /-
          🎉 no goals
        -/
            (mt (congr_arg (· % 2)) (by simp [natDegree_eq_of_degree_eq_some hf2, hR])))
          (card_image_polynomial_eval (by rw [degree_neg, hg2]; decide)))
      _ = 2 * #((univ.image fun x : R => eval x f) ∪ univ.image fun x : R => eval x (-g)) := by
        rw [card_union_of_disjoint hd]
        simp [natDegree_eq_of_degree_eq_some hf2, natDegree_eq_of_degree_eq_some hg2, mul_add]


theorem prod_univ_units_id_eq_neg_one [CommRing K] [IsDomain K] [Fintype Kˣ] :
    ∏ x : Kˣ, x = (-1 : Kˣ) := by
  classical
    have : (∏ x ∈ (@univ Kˣ _).erase (-1), x) = 1 :=
      prod_involution (fun x _ => x⁻¹) (by simp)
        (fun a => by simp +contextual [Units.inv_eq_self_iff])
        (fun a => by simp [@inv_eq_iff_eq_inv _ _ a]) (by simp)
    rw [← insert_erase (mem_univ (-1 : Kˣ)), prod_insert (not_mem_erase _ _), this, mul_one]


theorem card_cast_subgroup_card_ne_zero [Ring K] [NoZeroDivisors K] [Nontrivial K]
    (G : Subgroup Kˣ) [Fintype G] : (Fintype.card G : K) ≠ 0 := by
  /-
    K : Type u_1
    inst✝³ : Ring K
    inst✝² : NoZeroDivisors K
    inst✝¹ : Nontrivial K
    G : Subgroup (Units K)
    inst✝ : Fintype (Subtype fun x => Membership.mem G x)
    ⊢ Ne (↑(Fintype.card (Subtype fun x => Membership.mem G x))) 0
  -/
  let n := Fintype.card G
  /-
    K : Type u_1
    inst✝³ : Ring K
    inst✝² : NoZeroDivisors K
    inst✝¹ : Nontrivial K
    G : Subgroup (Units K)
    inst✝ : Fintype (Subtype fun x => Membership.mem G x)
    n : Nat := Fintype.card (Subtype fun x => Membership.mem G x)
    ⊢ Ne (↑(Fintype.card (Subtype fun x => Membership.mem G x))) 0
  -/
  intro nzero
  /-
    K : Type u_1
    inst✝³ : Ring K
    inst✝² : NoZeroDivisors K
    inst✝¹ : Nontrivial K
    G : Subgroup (Units K)
    inst✝ : Fintype (Subtype fun x => Membership.mem G x)
    n : Nat := Fintype.card (Subtype fun x => Membership.mem G x)
    nzero : Eq (↑(Fintype.card (Subtype fun x => Membership.mem G x))) 0
    ⊢ False
  -/
  have ⟨p, char_p⟩ := CharP.exists K
  /-
    K : Type u_1
    inst✝³ : Ring K
    inst✝² : NoZeroDivisors K
    inst✝¹ : Nontrivial K
    G : Subgroup (Units K)
    inst✝ : Fintype (Subtype fun x => Membership.mem G x)
    n : Nat := Fintype.card (Subtype fun x => Membership.mem G x)
    nzero : Eq (↑(Fintype.card (Subtype fun x => Membership.mem G x))) 0
    p : Nat
    char_p : CharP K p
    ⊢ False
  -/
  have hd : p ∣ n := (CharP.cast_eq_zero_iff K p n).mp nzero
  cases CharP.char_is_prime_or_zero K p with
  | inr pzero =>
    exact (Fintype.card_pos).ne' <| Nat.eq_zero_of_zero_dvd <| pzero ▸ hd
  | inl pprime =>
    have fact_pprime := Fact.mk pprime
    -- G has an element x of order p by Cauchy's theorem
    have ⟨x, hx⟩ := exists_prime_orderOf_dvd_card p hd
    -- F has an element u (= ↑↑x) of order p
    let u := ((x : Kˣ) : K)
    have hu : orderOf u = p := by rwa [orderOf_units, Subgroup.orderOf_coe]
    -- u ^ p = 1 implies (u - 1) ^ p = 0 and hence u = 1 ...
    have h : u = 1 := by
      rw [← sub_left_inj, sub_self 1]
      apply pow_eq_zero (n := p)
      rw [sub_pow_char_of_commute, one_pow, ← hu, pow_orderOf_eq_one, sub_self]
      exact Commute.one_right u
    -- ... meaning x didn't have order p after all, contradiction
    apply pprime.one_lt.ne
    rw [← hu, h, orderOf_one]


/-- The sum of a nontrivial subgroup of the units of a field is zero. -/
theorem sum_subgroup_units_eq_zero [Ring K] [NoZeroDivisors K]
    {G : Subgroup Kˣ} [Fintype G] (hg : G ≠ ⊥) :
    ∑ x : G, (x.val : K) = 0 := by
  /-
    K : Type u_1
    inst✝² : Ring K
    inst✝¹ : NoZeroDivisors K
    G : Subgroup (Units K)
    inst✝ : Fintype (Subtype fun x => Membership.mem G x)
    hg : Ne G Bot.bot
    ⊢ Eq (Finset.univ.sum fun x => ↑↑x) 0
  -/
  rw [Subgroup.ne_bot_iff_exists_ne_one] at hg
  /-
    K : Type u_1
    inst✝² : Ring K
    inst✝¹ : NoZeroDivisors K
    G : Subgroup (Units K)
    inst✝ : Fintype (Subtype fun x => Membership.mem G x)
    hg : Exists fun a => Ne a 1
    ⊢ Eq (Finset.univ.sum fun x => ↑↑x) 0
  -/
  rcases hg with ⟨a, ha⟩
  -- The action of a on G as an embedding
  /-
    case intro
    K : Type u_1
    inst✝² : Ring K
    inst✝¹ : NoZeroDivisors K
    G : Subgroup (Units K)
    inst✝ : Fintype (Subtype fun x => Membership.mem G x)
    a : Subtype fun x => Membership.mem G x
    ha : Ne a 1
    ⊢ Eq (Finset.univ.sum fun x => ↑↑x) 0
  -/
  let a_mul_emb : G ↪ G := mulLeftEmbedding a
  -- ... and leaves G unchanged
  /-
    case intro
    K : Type u_1
    inst✝² : Ring K
    inst✝¹ : NoZeroDivisors K
    G : Subgroup (Units K)
    inst✝ : Fintype (Subtype fun x => Membership.mem G x)
    a : Subtype fun x => Membership.mem G x
    ha : Ne a 1
    a_mul_emb : Function.Embedding (Subtype fun x => Membership.mem G x) (Subtype  …
    ⊢ Eq (Finset.univ.sum fun x => ↑↑x) 0
  -/
  have h_unchanged : Finset.univ.map a_mul_emb = Finset.univ := by simp
  -- Therefore the sum of x over a G is the sum of a x over G
  /-
    case intro
    K : Type u_1
    inst✝² : Ring K
    inst✝¹ : NoZeroDivisors K
    G : Subgroup (Units K)
    inst✝ : Fintype (Subtype fun x => Membership.mem G x)
    a : Subtype fun x => Membership.mem G x
    ha : Ne a 1
    a_mul_emb : Function.Embedding (Subtype fun x => Membership.mem G x) (Subtype  …
    h_unchanged : Eq (Finset.map a_mul_emb Finset.univ) Finset.univ
    ⊢ Eq (Finset.univ.sum fun x => ↑↑x) 0
  -/
  have h_sum_map := Finset.univ.sum_map a_mul_emb fun x => ((x : Kˣ) : K)
  -- ... and the former is the sum of x over G.
  -- By algebraic manipulation, we have Σ G, x = ∑ G, a x = a ∑ G, x
  simp only [h_unchanged, mulLeftEmbedding_apply, Subgroup.coe_mul, Units.val_mul, ← mul_sum,
    a_mul_emb] at h_sum_map
  -- thus one of (a - 1) or ∑ G, x is zero
  have hzero : (((a : Kˣ) : K) - 1) = 0 ∨ ∑ x : ↥G, ((x : Kˣ) : K) = 0 := by
    rw [← mul_eq_zero, sub_mul, ← h_sum_map, one_mul, sub_self]
  /-
    case intro
    K : Type u_1
    inst✝² : Ring K
    inst✝¹ : NoZeroDivisors K
    G : Subgroup (Units K)
    inst✝ : Fintype (Subtype fun x => Membership.mem G x)
    a : Subtype fun x => Membership.mem G x
    ha : Ne a 1
    a_mul_emb : Function.Embedding (Subtype fun x => Membership.mem G x) (Subtype  …
    h_unchanged : Eq (Finset.map a_mul_emb Finset.univ) Finset.univ
    h_sum_map : Eq (Finset.univ.sum fun x => ↑↑x) (HMul.hMul (↑↑a) (Finset.univ.su …
    hzero : Or (Eq (HSub.hSub (↑↑a) 1) 0) (Eq (Finset.univ.sum fun x => ↑↑x) 0)
    ⊢ Eq (Finset.univ.sum fun x => ↑↑x) 0
  -/
  apply Or.resolve_left hzero
  /-
    case intro
    K : Type u_1
    inst✝² : Ring K
    inst✝¹ : NoZeroDivisors K
    G : Subgroup (Units K)
    inst✝ : Fintype (Subtype fun x => Membership.mem G x)
    a : Subtype fun x => Membership.mem G x
    ha : Ne a 1
    a_mul_emb : Function.Embedding (Subtype fun x => Membership.mem G x) (Subtype  …
    h_unchanged : Eq (Finset.map a_mul_emb Finset.univ) Finset.univ
    h_sum_map : Eq (Finset.univ.sum fun x => ↑↑x) (HMul.hMul (↑↑a) (Finset.univ.su …
    hzero : Or (Eq (HSub.hSub (↑↑a) 1) 0) (Eq (Finset.univ.sum fun x => ↑↑x) 0)
    ⊢ Not (Eq (HSub.hSub (↑↑a) 1) 0)
  -/
  contrapose! ha
  /-
    case intro
    K : Type u_1
    inst✝² : Ring K
    inst✝¹ : NoZeroDivisors K
    G : Subgroup (Units K)
    inst✝ : Fintype (Subtype fun x => Membership.mem G x)
    a : Subtype fun x => Membership.mem G x
    a_mul_emb : Function.Embedding (Subtype fun x => Membership.mem G x) (Subtype  …
    h_unchanged : Eq (Finset.map a_mul_emb Finset.univ) Finset.univ
    h_sum_map : Eq (Finset.univ.sum fun x => ↑↑x) (HMul.hMul (↑↑a) (Finset.univ.su …
    hzero : Or (Eq (HSub.hSub (↑↑a) 1) 0) (Eq (Finset.univ.sum fun x => ↑↑x) 0)
    ha : Eq (HSub.hSub (↑↑a) 1) 0
    ⊢ Eq a 1
  -/
  ext
  /-
    case intro.a.a
    K : Type u_1
    inst✝² : Ring K
    inst✝¹ : NoZeroDivisors K
    G : Subgroup (Units K)
    inst✝ : Fintype (Subtype fun x => Membership.mem G x)
    a : Subtype fun x => Membership.mem G x
    a_mul_emb : Function.Embedding (Subtype fun x => Membership.mem G x) (Subtype  …
    h_unchanged : Eq (Finset.map a_mul_emb Finset.univ) Finset.univ
    h_sum_map : Eq (Finset.univ.sum fun x => ↑↑x) (HMul.hMul (↑↑a) (Finset.univ.su …
    hzero : Or (Eq (HSub.hSub (↑↑a) 1) 0) (Eq (Finset.univ.sum fun x => ↑↑x) 0)
    ha : Eq (HSub.hSub (↑↑a) 1) 0
    ⊢ Eq ↑↑a ↑↑1
  -/
  rwa [← sub_eq_zero]
  /-
    🎉 no goals
  -/


/-- The sum of a subgroup of the units of a field is 1 if the subgroup is trivial and 1 otherwise -/
@[simp]
theorem sum_subgroup_units [Ring K] [NoZeroDivisors K]
    {G : Subgroup Kˣ} [Fintype G] [Decidable (G = ⊥)] :
    ∑ x : G, (x.val : K) = if G = ⊥ then 1 else 0 := by
  /-
    K : Type u_1
    inst✝³ : Ring K
    inst✝² : NoZeroDivisors K
    G : Subgroup (Units K)
    inst✝¹ : Fintype (Subtype fun x => Membership.mem G x)
    inst✝ : Decidable (Eq G Bot.bot)
    ⊢ Eq (Finset.univ.sum fun x => ↑↑x) (ite (Eq G Bot.bot) 1 0)
  -/
  by_cases G_bot : G = ⊥
    /-
      case pos
      K : Type u_1
      inst✝³ : Ring K
      inst✝² : NoZeroDivisors K
      G : Subgroup (Units K)
      inst✝¹ : Fintype (Subtype fun x => Membership.mem G x)
      inst✝ : Decidable (Eq G Bot.bot)
      G_bot : Eq G Bot.bot
      ⊢ Eq (Finset.univ.sum fun x => ↑↑x) (ite (Eq G Bot.bot) 1 0)
    -/
  · subst G_bot
    /-
      case pos
      K : Type u_1
      inst✝³ : Ring K
      inst✝² : NoZeroDivisors K
      inst✝¹ : Fintype (Subtype fun x => Membership.mem Bot.bot x)
      inst✝ : Decidable (Eq Bot.bot Bot.bot)
      ⊢ Eq (Finset.univ.sum fun x => ↑↑x) (ite (Eq Bot.bot Bot.bot) 1 0)
    -/
    simp only [univ_unique, sum_singleton, ↓reduceIte, Units.val_eq_one, OneMemClass.coe_eq_one]
    /-
      case pos
      K : Type u_1
      inst✝³ : Ring K
      inst✝² : NoZeroDivisors K
      inst✝¹ : Fintype (Subtype fun x => Membership.mem Bot.bot x)
      inst✝ : Decidable (Eq Bot.bot Bot.bot)
      ⊢ Eq Inhabited.default 1
    -/
    rw [Set.default_coe_singleton]
    /-
      case pos
      K : Type u_1
      inst✝³ : Ring K
      inst✝² : NoZeroDivisors K
      inst✝¹ : Fintype (Subtype fun x => Membership.mem Bot.bot x)
      inst✝ : Decidable (Eq Bot.bot Bot.bot)
      ⊢ Eq ⟨1, ⋯⟩ 1
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      K : Type u_1
      inst✝³ : Ring K
      inst✝² : NoZeroDivisors K
      G : Subgroup (Units K)
      inst✝¹ : Fintype (Subtype fun x => Membership.mem G x)
      inst✝ : Decidable (Eq G Bot.bot)
      G_bot : Not (Eq G Bot.bot)
      ⊢ Eq (Finset.univ.sum fun x => ↑↑x) (ite (Eq G Bot.bot) 1 0)
    -/
  · simp only [G_bot, ite_false]
    /-
      case neg
      K : Type u_1
      inst✝³ : Ring K
      inst✝² : NoZeroDivisors K
      G : Subgroup (Units K)
      inst✝¹ : Fintype (Subtype fun x => Membership.mem G x)
      inst✝ : Decidable (Eq G Bot.bot)
      G_bot : Not (Eq G Bot.bot)
      ⊢ Eq (Finset.univ.sum fun x => ↑↑x) 0
    -/
    exact sum_subgroup_units_eq_zero G_bot
    /-
      🎉 no goals
    -/


@[simp]
theorem sum_subgroup_pow_eq_zero [CommRing K] [NoZeroDivisors K]
    {G : Subgroup Kˣ} [Fintype G] {k : ℕ} (k_pos : k ≠ 0) (k_lt_card_G : k < Fintype.card G) :
    ∑ x : G, ((x : Kˣ) : K) ^ k = 0 := by
  /-
    K : Type u_1
    inst✝² : CommRing K
    inst✝¹ : NoZeroDivisors K
    G : Subgroup (Units K)
    inst✝ : Fintype (Subtype fun x => Membership.mem G x)
    k : Nat
    k_pos : Ne k 0
    k_lt_card_G : LT.lt k (Fintype.card (Subtype fun x => Membership.mem G x))
    ⊢ Eq (Finset.univ.sum fun x => HPow.hPow (↑↑x) k) 0
  -/
  rw [← Nat.card_eq_fintype_card] at k_lt_card_G
  /-
    K : Type u_1
    inst✝² : CommRing K
    inst✝¹ : NoZeroDivisors K
    G : Subgroup (Units K)
    inst✝ : Fintype (Subtype fun x => Membership.mem G x)
    k : Nat
    k_pos : Ne k 0
    k_lt_card_G : LT.lt k (Nat.card (Subtype fun x => Membership.mem G x))
    ⊢ Eq (Finset.univ.sum fun x => HPow.hPow (↑↑x) k) 0
  -/
  nontriviality K
  /-
    K : Type u_1
    inst✝² : CommRing K
    inst✝¹ : NoZeroDivisors K
    G : Subgroup (Units K)
    inst✝ : Fintype (Subtype fun x => Membership.mem G x)
    k : Nat
    k_pos : Ne k 0
    k_lt_card_G : LT.lt k (Nat.card (Subtype fun x => Membership.mem G x))
    a✝ : Nontrivial K
    ⊢ Eq (Finset.univ.sum fun x => HPow.hPow (↑↑x) k) 0
  -/
  have := NoZeroDivisors.to_isDomain K
  /-
    K : Type u_1
    inst✝² : CommRing K
    inst✝¹ : NoZeroDivisors K
    G : Subgroup (Units K)
    inst✝ : Fintype (Subtype fun x => Membership.mem G x)
    k : Nat
    k_pos : Ne k 0
    k_lt_card_G : LT.lt k (Nat.card (Subtype fun x => Membership.mem G x))
    a✝ : Nontrivial K
    this : IsDomain K
    ⊢ Eq (Finset.univ.sum fun x => HPow.hPow (↑↑x) k) 0
  -/
  rcases (exists_pow_ne_one_of_isCyclic k_pos k_lt_card_G) with ⟨a, ha⟩
  /-
    case intro
    K : Type u_1
    inst✝² : CommRing K
    inst✝¹ : NoZeroDivisors K
    G : Subgroup (Units K)
    inst✝ : Fintype (Subtype fun x => Membership.mem G x)
    k : Nat
    k_pos : Ne k 0
    k_lt_card_G : LT.lt k (Nat.card (Subtype fun x => Membership.mem G x))
    a✝ : Nontrivial K
    this : IsDomain K
    a : Subtype fun x => Membership.mem G x
    ha : Ne (HPow.hPow a k) 1
    ⊢ Eq (Finset.univ.sum fun x => HPow.hPow (↑↑x) k) 0
  -/
  rw [Finset.sum_eq_multiset_sum]
  have h_multiset_map :
    Finset.univ.val.map (fun x : G => ((x : Kˣ) : K) ^ k) =
      Finset.univ.val.map (fun x : G => ((x : Kˣ) : K) ^ k * ((a : Kˣ) : K) ^ k) := by
    simp_rw [← mul_pow]
    have as_comp :
      (fun x : ↥G => (((x : Kˣ) : K) * ((a : Kˣ) : K)) ^ k)
        = (fun x : ↥G => ((x : Kˣ) : K) ^ k) ∘ fun x : ↥G => x * a := by
      funext x
      simp only [Function.comp_apply, Subgroup.coe_mul, Units.val_mul]
    rw [as_comp, ← Multiset.map_map]
    congr
    rw [eq_comm]
    exact Multiset.map_univ_val_equiv (Equiv.mulRight a)
  have h_multiset_map_sum : (Multiset.map (fun x : G => ((x : Kˣ) : K) ^ k) Finset.univ.val).sum =
    (Multiset.map (fun x : G => ((x : Kˣ) : K) ^ k * ((a : Kˣ) : K) ^ k) Finset.univ.val).sum := by
    rw [h_multiset_map]
  /-
    case intro
    K : Type u_1
    inst✝² : CommRing K
    inst✝¹ : NoZeroDivisors K
    G : Subgroup (Units K)
    inst✝ : Fintype (Subtype fun x => Membership.mem G x)
    k : Nat
    k_pos : Ne k 0
    k_lt_card_G : LT.lt k (Nat.card (Subtype fun x => Membership.mem G x))
    a✝ : Nontrivial K
    this : IsDomain K
    a : Subtype fun x => Membership.mem G x
    ha : Ne (HPow.hPow a k) 1
    h_multiset_map : Eq (Multiset.map (fun x => HPow.hPow (↑↑x) k) Finset.univ.val …
    h_multiset_map_sum : Eq (Multiset.map (fun x => HPow.hPow (↑↑x) k) Finset.univ …
    ⊢ Eq (Multiset.map (fun x => HPow.hPow (↑↑x) k) Finset.univ.val).sum 0
  -/
  rw [Multiset.sum_map_mul_right] at h_multiset_map_sum
  have hzero : (((a : Kˣ) : K) ^ k - 1 : K)
                  * (Multiset.map (fun i : G => (i.val : K) ^ k) Finset.univ.val).sum = 0 := by
    rw [sub_mul, mul_comm, ← h_multiset_map_sum, one_mul, sub_self]
  /-
    case intro
    K : Type u_1
    inst✝² : CommRing K
    inst✝¹ : NoZeroDivisors K
    G : Subgroup (Units K)
    inst✝ : Fintype (Subtype fun x => Membership.mem G x)
    k : Nat
    k_pos : Ne k 0
    k_lt_card_G : LT.lt k (Nat.card (Subtype fun x => Membership.mem G x))
    a✝ : Nontrivial K
    this : IsDomain K
    a : Subtype fun x => Membership.mem G x
    ha : Ne (HPow.hPow a k) 1
    h_multiset_map : Eq (Multiset.map (fun x => HPow.hPow (↑↑x) k) Finset.univ.val …
    h_multiset_map_sum : Eq (Multiset.map (fun x => HPow.hPow (↑↑x) k) Finset.univ …
    hzero : Eq (HMul.hMul (HSub.hSub (HPow.hPow (↑↑a) k) 1) (Multiset.map (fun i = …
    ⊢ Eq (Multiset.map (fun x => HPow.hPow (↑↑x) k) Finset.univ.val).sum 0
  -/
  rw [mul_eq_zero] at hzero
  /-
    case intro
    K : Type u_1
    inst✝² : CommRing K
    inst✝¹ : NoZeroDivisors K
    G : Subgroup (Units K)
    inst✝ : Fintype (Subtype fun x => Membership.mem G x)
    k : Nat
    k_pos : Ne k 0
    k_lt_card_G : LT.lt k (Nat.card (Subtype fun x => Membership.mem G x))
    a✝ : Nontrivial K
    this : IsDomain K
    a : Subtype fun x => Membership.mem G x
    ha : Ne (HPow.hPow a k) 1
    h_multiset_map : Eq (Multiset.map (fun x => HPow.hPow (↑↑x) k) Finset.univ.val …
    h_multiset_map_sum : Eq (Multiset.map (fun x => HPow.hPow (↑↑x) k) Finset.univ …
    hzero : Or (Eq (HSub.hSub (HPow.hPow (↑↑a) k) 1) 0) (Eq (Multiset.map (fun i = …
    ⊢ Eq (Multiset.map (fun x => HPow.hPow (↑↑x) k) Finset.univ.val).sum 0
  -/
  refine hzero.resolve_left fun h => ha ?_
  /-
    case intro
    K : Type u_1
    inst✝² : CommRing K
    inst✝¹ : NoZeroDivisors K
    G : Subgroup (Units K)
    inst✝ : Fintype (Subtype fun x => Membership.mem G x)
    k : Nat
    k_pos : Ne k 0
    k_lt_card_G : LT.lt k (Nat.card (Subtype fun x => Membership.mem G x))
    a✝ : Nontrivial K
    this : IsDomain K
    a : Subtype fun x => Membership.mem G x
    ha : Ne (HPow.hPow a k) 1
    h_multiset_map : Eq (Multiset.map (fun x => HPow.hPow (↑↑x) k) Finset.univ.val …
    h_multiset_map_sum : Eq (Multiset.map (fun x => HPow.hPow (↑↑x) k) Finset.univ …
    hzero : Or (Eq (HSub.hSub (HPow.hPow (↑↑a) k) 1) 0) (Eq (Multiset.map (fun i = …
    h : Eq (HSub.hSub (HPow.hPow (↑↑a) k) 1) 0
    ⊢ Eq (HPow.hPow a k) 1
  -/
  ext
  /-
    case intro.a.a
    K : Type u_1
    inst✝² : CommRing K
    inst✝¹ : NoZeroDivisors K
    G : Subgroup (Units K)
    inst✝ : Fintype (Subtype fun x => Membership.mem G x)
    k : Nat
    k_pos : Ne k 0
    k_lt_card_G : LT.lt k (Nat.card (Subtype fun x => Membership.mem G x))
    a✝ : Nontrivial K
    this : IsDomain K
    a : Subtype fun x => Membership.mem G x
    ha : Ne (HPow.hPow a k) 1
    h_multiset_map : Eq (Multiset.map (fun x => HPow.hPow (↑↑x) k) Finset.univ.val …
    h_multiset_map_sum : Eq (Multiset.map (fun x => HPow.hPow (↑↑x) k) Finset.univ …
    hzero : Or (Eq (HSub.hSub (HPow.hPow (↑↑a) k) 1) 0) (Eq (Multiset.map (fun i = …
    h : Eq (HSub.hSub (HPow.hPow (↑↑a) k) 1) 0
    ⊢ Eq ↑↑(HPow.hPow a k) ↑↑1
  -/
  rw [← sub_eq_zero]
  /-
    case intro.a.a
    K : Type u_1
    inst✝² : CommRing K
    inst✝¹ : NoZeroDivisors K
    G : Subgroup (Units K)
    inst✝ : Fintype (Subtype fun x => Membership.mem G x)
    k : Nat
    k_pos : Ne k 0
    k_lt_card_G : LT.lt k (Nat.card (Subtype fun x => Membership.mem G x))
    a✝ : Nontrivial K
    this : IsDomain K
    a : Subtype fun x => Membership.mem G x
    ha : Ne (HPow.hPow a k) 1
    h_multiset_map : Eq (Multiset.map (fun x => HPow.hPow (↑↑x) k) Finset.univ.val …
    h_multiset_map_sum : Eq (Multiset.map (fun x => HPow.hPow (↑↑x) k) Finset.univ …
    hzero : Or (Eq (HSub.hSub (HPow.hPow (↑↑a) k) 1) 0) (Eq (Multiset.map (fun i = …
    h : Eq (HSub.hSub (HPow.hPow (↑↑a) k) 1) 0
    ⊢ Eq (HSub.hSub ↑↑(HPow.hPow a k) ↑↑1) 0
  -/
  simp_rw [SubmonoidClass.coe_pow, Units.val_pow_eq_pow_val, OneMemClass.coe_one, Units.val_one, h]
  /-
    🎉 no goals
  -/


theorem pow_card_sub_one_eq_one (a : K) (ha : a ≠ 0) : a ^ (q - 1) = 1 := by
  calc
    a ^ (Fintype.card K - 1) = (Units.mk0 a ha ^ (Fintype.card K - 1) : Kˣ).1 := by
      rw [Units.val_pow_eq_pow_val, Units.val_mk0]
    _ = 1 := by
      classical
        rw [← Fintype.card_units, pow_card_eq_one]
        rfl


theorem pow_card (a : K) : a ^ q = a := by
  /-
    K : Type u_1
    inst✝¹ : GroupWithZero K
    inst✝ : Fintype K
    a : K
    ⊢ Eq (HPow.hPow a (Fintype.card K)) a
  -/
  by_cases h : a = 0; · rw [h]; apply zero_pow Fintype.card_ne_zero
                                /-
                                  🎉 no goals
                                -/
  rw [← Nat.succ_pred_eq_of_pos Fintype.card_pos, pow_succ, Nat.pred_eq_sub_one,
    pow_card_sub_one_eq_one a h, one_mul]


theorem pow_card_pow (n : ℕ) (a : K) : a ^ q ^ n = a := by
  induction n with
  | zero => simp
  | succ n ih => simp [pow_succ, pow_mul, ih, pow_card]


/-- The cardinality `q` is a power of the characteristic of `K`. -/
@[stacks 09HY "first part"]
theorem card (p : ℕ) [CharP K p] : ∃ n : ℕ+, Nat.Prime p ∧ q = p ^ (n : ℕ) := by
  /-
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : Fintype K
    p : Nat
    inst✝ : CharP K p
    ⊢ Exists fun n => And (Nat.Prime p) (Eq (Fintype.card K) (HPow.hPow p ↑n))
  -/
  haveI hp : Fact p.Prime := ⟨CharP.char_is_prime K p⟩
  /-
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : Fintype K
    p : Nat
    inst✝ : CharP K p
    hp : Fact (Nat.Prime p)
    ⊢ Exists fun n => And (Nat.Prime p) (Eq (Fintype.card K) (HPow.hPow p ↑n))
  -/
  letI : Module (ZMod p) K := { (ZMod.castHom dvd_rfl K : ZMod p →+* _).toModule with }
  /-
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : Fintype K
    p : Nat
    inst✝ : CharP K p
    hp : Fact (Nat.Prime p)
    this : Module (ZMod p) K :=
      let __src := (ZMod.castHom ⋯ K).toModule;
      Module.mk ⋯ ⋯
    ⊢ Exists fun n => And (Nat.Prime p) (Eq (Fintype.card K) (HPow.hPow p ↑n))
  -/
  obtain ⟨n, h⟩ := VectorSpace.card_fintype (ZMod p) K
  /-
    case intro
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : Fintype K
    p : Nat
    inst✝ : CharP K p
    hp : Fact (Nat.Prime p)
    this : Module (ZMod p) K :=
      let __src := (ZMod.castHom ⋯ K).toModule;
      Module.mk ⋯ ⋯
    n : Nat
    h : Eq (Fintype.card K) (HPow.hPow (Fintype.card (ZMod p)) n)
    ⊢ Exists fun n => And (Nat.Prime p) (Eq (Fintype.card K) (HPow.hPow p ↑n))
  -/
  rw [ZMod.card] at h
  /-
    case intro
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : Fintype K
    p : Nat
    inst✝ : CharP K p
    hp : Fact (Nat.Prime p)
    this : Module (ZMod p) K :=
      let __src := (ZMod.castHom ⋯ K).toModule;
      Module.mk ⋯ ⋯
    n : Nat
    h : Eq (Fintype.card K) (HPow.hPow p n)
    ⊢ Exists fun n => And (Nat.Prime p) (Eq (Fintype.card K) (HPow.hPow p ↑n))
  -/
  refine ⟨⟨n, ?_⟩, hp.1, h⟩
  /-
    case intro
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : Fintype K
    p : Nat
    inst✝ : CharP K p
    hp : Fact (Nat.Prime p)
    this : Module (ZMod p) K :=
      let __src := (ZMod.castHom ⋯ K).toModule;
      Module.mk ⋯ ⋯
    n : Nat
    h : Eq (Fintype.card K) (HPow.hPow p n)
    ⊢ LT.lt 0 n
  -/
  apply Or.resolve_left (Nat.eq_zero_or_pos n)
  /-
    case intro
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : Fintype K
    p : Nat
    inst✝ : CharP K p
    hp : Fact (Nat.Prime p)
    this : Module (ZMod p) K :=
      let __src := (ZMod.castHom ⋯ K).toModule;
      Module.mk ⋯ ⋯
    n : Nat
    h : Eq (Fintype.card K) (HPow.hPow p n)
    ⊢ Not (Eq n 0)
  -/
  rintro rfl
  /-
    case intro
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : Fintype K
    p : Nat
    inst✝ : CharP K p
    hp : Fact (Nat.Prime p)
    this : Module (ZMod p) K :=
      let __src := (ZMod.castHom ⋯ K).toModule;
      Module.mk ⋯ ⋯
    h : Eq (Fintype.card K) (HPow.hPow p 0)
    ⊢ False
  -/
  rw [pow_zero] at h
  /-
    case intro
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : Fintype K
    p : Nat
    inst✝ : CharP K p
    hp : Fact (Nat.Prime p)
    this : Module (ZMod p) K :=
      let __src := (ZMod.castHom ⋯ K).toModule;
      Module.mk ⋯ ⋯
    h : Eq (Fintype.card K) 1
    ⊢ False
  -/
  have : (0 : K) = 1 := by apply Fintype.card_le_one_iff.mp (le_of_eq h)
  /-
    case intro
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : Fintype K
    p : Nat
    inst✝ : CharP K p
    hp : Fact (Nat.Prime p)
    this✝ : Module (ZMod p) K :=
      let __src := (ZMod.castHom ⋯ K).toModule;
      Module.mk ⋯ ⋯
    h : Eq (Fintype.card K) 1
    this : Eq 0 1
    ⊢ False
  -/
  exact absurd this zero_ne_one
  /-
    🎉 no goals
  -/

-- this statement doesn't use `q` because we want `K` to be an explicit parameter

theorem card' : ∃ (p : ℕ) (n : ℕ+), Nat.Prime p ∧ Fintype.card K = p ^ (n : ℕ) :=
  let ⟨p, hc⟩ := CharP.exists K
  ⟨p, @FiniteField.card K _ _ p hc⟩

-- Porting note: this was a `simp` lemma with a 5 lines proof.

theorem cast_card_eq_zero : (q : K) = 0 := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : Fintype K
    ⊢ Eq (↑(Fintype.card K)) 0
  -/
  simp
  /-
    🎉 no goals
  -/


theorem forall_pow_eq_one_iff (i : ℕ) : (∀ x : Kˣ, x ^ i = 1) ↔ q - 1 ∣ i := by
  classical
    obtain ⟨x, hx⟩ := IsCyclic.exists_generator (α := Kˣ)
    rw [← Nat.card_eq_fintype_card, ← Nat.card_units, ← orderOf_eq_card_of_forall_mem_zpowers hx,
      orderOf_dvd_iff_pow_eq_one]
    constructor
    · intro h; apply h
    · intro h y
      simp_rw [← mem_powers_iff_mem_zpowers] at hx
      rcases hx y with ⟨j, rfl⟩
      rw [← pow_mul, mul_comm, pow_mul, h, one_pow]


/-- The sum of `x ^ i` as `x` ranges over the units of a finite field of cardinality `q`
is equal to `0` unless `(q - 1) ∣ i`, in which case the sum is `q - 1`. -/
theorem sum_pow_units [DecidableEq K] (i : ℕ) :
    (∑ x : Kˣ, (x ^ i : K)) = if q - 1 ∣ i then -1 else 0 := by
  let φ : Kˣ →* K :=
    { toFun := fun x => x ^ i
      map_one' := by simp
      map_mul' := by intros; simp [mul_pow] }
  /-
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : Fintype K
    inst✝ : DecidableEq K
    i : Nat
    φ : MonoidHom (Units K) K := { toFun := fun x => HPow.hPow (↑x) i, map_one' := …
    ⊢ Eq (Finset.univ.sum fun x => HPow.hPow (↑x) i) (ite (Dvd.dvd (HSub.hSub (Fin …
  -/
  have : Decidable (φ = 1) := by classical infer_instance
  calc (∑ x : Kˣ, φ x) = if φ = 1 then Fintype.card Kˣ else 0 := sum_hom_units φ
      _ = if q - 1 ∣ i then -1 else 0 := by
        suffices q - 1 ∣ i ↔ φ = 1 by
          simp only [this]
          split_ifs; swap
          · exact Nat.cast_zero
          · rw [Fintype.card_units, Nat.cast_sub,
              cast_card_eq_zero, Nat.cast_one, zero_sub]
            show 1 ≤ q; exact Fintype.card_pos_iff.mpr ⟨0⟩
        rw [← forall_pow_eq_one_iff, DFunLike.ext_iff]
        apply forall_congr'; intro x; simp [φ, Units.ext_iff]


/-- The sum of `x ^ i` as `x` ranges over a finite field of cardinality `q`
is equal to `0` if `i < q - 1`. -/
theorem sum_pow_lt_card_sub_one (i : ℕ) (h : i < q - 1) : ∑ x : K, x ^ i = 0 := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : Fintype K
    i : Nat
    h : LT.lt i (HSub.hSub (Fintype.card K) 1)
    ⊢ Eq (Finset.univ.sum fun x => HPow.hPow x i) 0
  -/
  by_cases hi : i = 0
    /-
      case pos
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : Fintype K
      i : Nat
      h : LT.lt i (HSub.hSub (Fintype.card K) 1)
      hi : Eq i 0
      ⊢ Eq (Finset.univ.sum fun x => HPow.hPow x i) 0
    -/
  · simp only [hi, nsmul_one, sum_const, pow_zero, card_univ, cast_card_eq_zero]
    /-
      🎉 no goals
    -/
  classical
    have hiq : ¬q - 1 ∣ i := by contrapose! h; exact Nat.le_of_dvd (Nat.pos_of_ne_zero hi) h
    let φ : Kˣ ↪ K := ⟨fun x ↦ x, Units.ext⟩
    have : univ.map φ = univ \ {0} := by
      ext x
      simpa only [mem_map, mem_univ, Function.Embedding.coeFn_mk, true_and, mem_sdiff,
        mem_singleton, φ] using isUnit_iff_ne_zero
    calc
      ∑ x : K, x ^ i = ∑ x ∈ univ \ {(0 : K)}, x ^ i := by
        rw [← sum_sdiff ({0} : Finset K).subset_univ, sum_singleton, zero_pow hi, add_zero]
      _ = ∑ x : Kˣ, (x ^ i : K) := by simp [φ, ← this, univ.sum_map φ]
      _ = 0 := by rw [sum_pow_units K i, if_neg]; exact hiq


theorem X_pow_card_sub_X_natDegree_eq (hp : 1 < p) : (X ^ p - X : K'[X]).natDegree = p := by
  have h1 : (X : K'[X]).degree < (X ^ p : K'[X]).degree := by
    rw [degree_X_pow, degree_X]
    exact mod_cast hp
  /-
    K' : Type u_3
    inst✝ : Field K'
    p : Nat
    hp : LT.lt 1 p
    h1 : LT.lt Polynomial.X.degree (HPow.hPow Polynomial.X p).degree
    ⊢ Eq (HSub.hSub (HPow.hPow Polynomial.X p) Polynomial.X).natDegree p
  -/
  rw [natDegree_eq_of_degree_eq (degree_sub_eq_left_of_degree_lt h1), natDegree_X_pow]
  /-
    🎉 no goals
  -/


theorem X_pow_card_pow_sub_X_natDegree_eq (hn : n ≠ 0) (hp : 1 < p) :
    (X ^ p ^ n - X : K'[X]).natDegree = p ^ n :=
  X_pow_card_sub_X_natDegree_eq K' <| Nat.one_lt_pow hn hp


theorem X_pow_card_sub_X_ne_zero (hp : 1 < p) : (X ^ p - X : K'[X]) ≠ 0 :=
  ne_zero_of_natDegree_gt <|
    calc
      1 < _ := hp
      _ = _ := (X_pow_card_sub_X_natDegree_eq K' hp).symm


theorem X_pow_card_pow_sub_X_ne_zero (hn : n ≠ 0) (hp : 1 < p) : (X ^ p ^ n - X : K'[X]) ≠ 0 :=
  X_pow_card_sub_X_ne_zero K' <| Nat.one_lt_pow hn hp


theorem roots_X_pow_card_sub_X : roots (X ^ q - X : K[X]) = Finset.univ.val := by
  classical
    have aux : (X ^ q - X : K[X]) ≠ 0 := X_pow_card_sub_X_ne_zero K Fintype.one_lt_card
    have : (roots (X ^ q - X : K[X])).toFinset = Finset.univ := by
      rw [eq_univ_iff_forall]
      intro x
      rw [Multiset.mem_toFinset, mem_roots aux, IsRoot.def, eval_sub, eval_pow, eval_X,
        sub_eq_zero, pow_card]
    rw [← this, Multiset.toFinset_val, eq_comm, Multiset.dedup_eq_self]
    apply nodup_roots
    rw [separable_def]
    convert isCoprime_one_right.neg_right (R := K[X]) using 1
    rw [derivative_sub, derivative_X, derivative_X_pow, Nat.cast_card_eq_zero K, C_0,
      zero_mul, zero_sub]


theorem frobenius_pow {p : ℕ} [Fact p.Prime] [CharP K p] {n : ℕ} (hcard : q = p ^ n) :
    frobenius K p ^ n = 1 := by
  /-
    K : Type u_1
    inst✝³ : Field K
    inst✝² : Fintype K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    n : Nat
    hcard : Eq (Fintype.card K) (HPow.hPow p n)
    ⊢ Eq (HPow.hPow (frobenius K p) n) 1
  -/
  ext x; conv_rhs => rw [RingHom.one_def, RingHom.id_apply, ← pow_card x, hcard]
  /-
    case a
    K : Type u_1
    inst✝³ : Field K
    inst✝² : Fintype K
    p : Nat
    inst✝¹ : Fact (Nat.Prime p)
    inst✝ : CharP K p
    n : Nat
    hcard : Eq (Fintype.card K) (HPow.hPow p n)
    x : K
    ⊢ Eq ((HPow.hPow (frobenius K p) n) x) (HPow.hPow x (HPow.hPow p n))
  -/
  clear hcard
  induction n with
  | zero => simp
  | succ n hn =>
    rw [pow_succ', pow_succ, pow_mul, RingHom.mul_def, RingHom.comp_apply, frobenius_def, hn]


theorem expand_card (f : K[X]) : expand K q f = f ^ q := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : Fintype K
    f : Polynomial K
    ⊢ Eq ((Polynomial.expand K (Fintype.card K)) f) (HPow.hPow f (Fintype.card K))
  -/
  cases' CharP.exists K with p hp
  /-
    case intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : Fintype K
    f : Polynomial K
    p : Nat
    hp : CharP K p
    ⊢ Eq ((Polynomial.expand K (Fintype.card K)) f) (HPow.hPow f (Fintype.card K))
  -/
  letI := hp
  /-
    case intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : Fintype K
    f : Polynomial K
    p : Nat
    hp : CharP K p
    this : CharP K p := hp
    ⊢ Eq ((Polynomial.expand K (Fintype.card K)) f) (HPow.hPow f (Fintype.card K))
  -/
  rcases FiniteField.card K p with ⟨⟨n, npos⟩, ⟨hp, hn⟩⟩
  /-
    case intro.intro.mk.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : Fintype K
    f : Polynomial K
    p : Nat
    hp✝ : CharP K p
    this : CharP K p := hp✝
    n : Nat
    npos : LT.lt 0 n
    hp : Nat.Prime p
    hn : Eq (Fintype.card K) (HPow.hPow p ↑⟨n, npos⟩)
    ⊢ Eq ((Polynomial.expand K (Fintype.card K)) f) (HPow.hPow f (Fintype.card K))
  -/
  haveI : Fact p.Prime := ⟨hp⟩
  /-
    case intro.intro.mk.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : Fintype K
    f : Polynomial K
    p : Nat
    hp✝ : CharP K p
    this✝ : CharP K p := hp✝
    n : Nat
    npos : LT.lt 0 n
    hp : Nat.Prime p
    hn : Eq (Fintype.card K) (HPow.hPow p ↑⟨n, npos⟩)
    this : Fact (Nat.Prime p)
    ⊢ Eq ((Polynomial.expand K (Fintype.card K)) f) (HPow.hPow f (Fintype.card K))
  -/
  dsimp at hn
  /-
    case intro.intro.mk.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : Fintype K
    f : Polynomial K
    p : Nat
    hp✝ : CharP K p
    this✝ : CharP K p := hp✝
    n : Nat
    npos : LT.lt 0 n
    hp : Nat.Prime p
    hn : Eq (Fintype.card K) (HPow.hPow p n)
    this : Fact (Nat.Prime p)
    ⊢ Eq ((Polynomial.expand K (Fintype.card K)) f) (HPow.hPow f (Fintype.card K))
  -/
  rw [hn, ← map_expand_pow_char, frobenius_pow hn, RingHom.one_def, map_id]
  /-
    🎉 no goals
  -/


theorem sq_add_sq (p : ℕ) [hp : Fact p.Prime] (x : ZMod p) : ∃ a b : ZMod p, a ^ 2 + b ^ 2 = x := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x : ZMod p
    ⊢ Exists fun a => Exists fun b => Eq (HAdd.hAdd (HPow.hPow a 2) (HPow.hPow b 2 …
  -/
  cases' hp.1.eq_two_or_odd with hp2 hp_odd
    /-
      case inl
      p : Nat
      hp : Fact (Nat.Prime p)
      x : ZMod p
      hp2 : Eq p 2
      ⊢ Exists fun a => Exists fun b => Eq (HAdd.hAdd (HPow.hPow a 2) (HPow.hPow b 2 …
    -/
  · subst p
    /-
      case inl
      hp : Fact (Nat.Prime 2)
      x : ZMod 2
      ⊢ Exists fun a => Exists fun b => Eq (HAdd.hAdd (HPow.hPow a 2) (HPow.hPow b 2 …
    -/
    change Fin 2 at x
    /-
      case inl
      hp : Fact (Nat.Prime 2)
      x : Fin 2
      ⊢ Exists fun a => Exists fun b => Eq (HAdd.hAdd (HPow.hPow a 2) (HPow.hPow b 2 …
    -/
    fin_cases x
      /-
        case inl.«0»
        hp : Fact (Nat.Prime 2)
        ⊢ Exists fun a => Exists fun b => Eq (HAdd.hAdd (HPow.hPow a 2) (HPow.hPow b 2 …
      -/
    · use 0; simp
             /-
               🎉 no goals
             -/
      /-
        case inl.«1»
        hp : Fact (Nat.Prime 2)
        ⊢ Exists fun a => Exists fun b => Eq (HAdd.hAdd (HPow.hPow a 2) (HPow.hPow b 2 …
      -/
    · use 0, 1; simp
                /-
                  🎉 no goals
                -/
  /-
    case inr
    p : Nat
    hp : Fact (Nat.Prime p)
    x : ZMod p
    hp_odd : Eq (HMod.hMod p 2) 1
    ⊢ Exists fun a => Exists fun b => Eq (HAdd.hAdd (HPow.hPow a 2) (HPow.hPow b 2 …
  -/
  let f : (ZMod p)[X] := X ^ 2
  /-
    case inr
    p : Nat
    hp : Fact (Nat.Prime p)
    x : ZMod p
    hp_odd : Eq (HMod.hMod p 2) 1
    f : Polynomial (ZMod p) := HPow.hPow Polynomial.X 2
    ⊢ Exists fun a => Exists fun b => Eq (HAdd.hAdd (HPow.hPow a 2) (HPow.hPow b 2 …
  -/
  let g : (ZMod p)[X] := X ^ 2 - C x
  obtain ⟨a, b, hab⟩ : ∃ a b, f.eval a + g.eval b = 0 :=
    @exists_root_sum_quadratic _ _ _ _ f g (degree_X_pow 2) (degree_X_pow_sub_C (by decide) _)
      (by rw [ZMod.card, hp_odd])
  /-
    case inr.intro.intro
    p : Nat
    hp : Fact (Nat.Prime p)
    x : ZMod p
    hp_odd : Eq (HMod.hMod p 2) 1
    f : Polynomial (ZMod p) := HPow.hPow Polynomial.X 2
    g : Polynomial (ZMod p) := HSub.hSub (HPow.hPow Polynomial.X 2) (Polynomial.C x)
    a b : ZMod p
    hab : Eq (HAdd.hAdd (Polynomial.eval a f) (Polynomial.eval b g)) 0
    ⊢ Exists fun a => Exists fun b => Eq (HAdd.hAdd (HPow.hPow a 2) (HPow.hPow b 2 …
  -/
  refine ⟨a, b, ?_⟩
  /-
    case inr.intro.intro
    p : Nat
    hp : Fact (Nat.Prime p)
    x : ZMod p
    hp_odd : Eq (HMod.hMod p 2) 1
    f : Polynomial (ZMod p) := HPow.hPow Polynomial.X 2
    g : Polynomial (ZMod p) := HSub.hSub (HPow.hPow Polynomial.X 2) (Polynomial.C x)
    a b : ZMod p
    hab : Eq (HAdd.hAdd (Polynomial.eval a f) (Polynomial.eval b g)) 0
    ⊢ Eq (HAdd.hAdd (HPow.hPow a 2) (HPow.hPow b 2)) x
  -/
  rw [← sub_eq_zero]
  /-
    case inr.intro.intro
    p : Nat
    hp : Fact (Nat.Prime p)
    x : ZMod p
    hp_odd : Eq (HMod.hMod p 2) 1
    f : Polynomial (ZMod p) := HPow.hPow Polynomial.X 2
    g : Polynomial (ZMod p) := HSub.hSub (HPow.hPow Polynomial.X 2) (Polynomial.C x)
    a b : ZMod p
    hab : Eq (HAdd.hAdd (Polynomial.eval a f) (Polynomial.eval b g)) 0
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HPow.hPow a 2) (HPow.hPow b 2)) x) 0
  -/
  simpa only [f, g, eval_C, eval_X, eval_pow, eval_sub, ← add_sub_assoc] using hab
  /-
    🎉 no goals
  -/


/-- If `p` is a prime natural number and `x` is an integer number, then there exist natural numbers
`a ≤ p / 2` and `b ≤ p / 2` such that `a ^ 2 + b ^ 2 ≡ x [ZMOD p]`. This is a version of
`ZMod.sq_add_sq` with estimates on `a` and `b`. -/
theorem Nat.sq_add_sq_zmodEq (p : ℕ) [Fact p.Prime] (x : ℤ) :
    ∃ a b : ℕ, a ≤ p / 2 ∧ b ≤ p / 2 ∧ (a : ℤ) ^ 2 + (b : ℤ) ^ 2 ≡ x [ZMOD p] := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    x : Int
    ⊢ Exists fun a => Exists fun b => And (LE.le a (HDiv.hDiv p 2)) (And (LE.le b  …
  -/
  rcases ZMod.sq_add_sq p x with ⟨a, b, hx⟩
  refine ⟨a.valMinAbs.natAbs, b.valMinAbs.natAbs, ZMod.natAbs_valMinAbs_le _,
    ZMod.natAbs_valMinAbs_le _, ?_⟩
  /-
    case intro.intro
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    x : Int
    a b : ZMod p
    hx : Eq (HAdd.hAdd (HPow.hPow a 2) (HPow.hPow b 2)) ↑x
    ⊢ (↑p).ModEq (HAdd.hAdd (HPow.hPow (↑a.valMinAbs.natAbs) 2) (HPow.hPow (↑b.val …
  -/
  rw [← a.coe_valMinAbs, ← b.coe_valMinAbs] at hx
  /-
    case intro.intro
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    x : Int
    a b : ZMod p
    hx : Eq (HAdd.hAdd (HPow.hPow (↑a.valMinAbs) 2) (HPow.hPow (↑b.valMinAbs) 2)) ↑x
    ⊢ (↑p).ModEq (HAdd.hAdd (HPow.hPow (↑a.valMinAbs.natAbs) 2) (HPow.hPow (↑b.val …
  -/
  push_cast
  /-
    case intro.intro
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    x : Int
    a b : ZMod p
    hx : Eq (HAdd.hAdd (HPow.hPow (↑a.valMinAbs) 2) (HPow.hPow (↑b.valMinAbs) 2)) ↑x
    ⊢ (↑p).ModEq (HAdd.hAdd (HPow.hPow (abs a.valMinAbs) 2) (HPow.hPow (abs b.valM …
  -/
  rw [sq_abs, sq_abs, ← ZMod.intCast_eq_intCast_iff]
  /-
    case intro.intro
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    x : Int
    a b : ZMod p
    hx : Eq (HAdd.hAdd (HPow.hPow (↑a.valMinAbs) 2) (HPow.hPow (↑b.valMinAbs) 2)) ↑x
    ⊢ Eq ↑(HAdd.hAdd (HPow.hPow a.valMinAbs 2) (HPow.hPow b.valMinAbs 2)) ↑x
  -/
  exact mod_cast hx
  /-
    🎉 no goals
  -/


/-- If `p` is a prime natural number and `x` is a natural number, then there exist natural numbers
`a ≤ p / 2` and `b ≤ p / 2` such that `a ^ 2 + b ^ 2 ≡ x [MOD p]`. This is a version of
`ZMod.sq_add_sq` with estimates on `a` and `b`. -/
theorem Nat.sq_add_sq_modEq (p : ℕ) [Fact p.Prime] (x : ℕ) :
    ∃ a b : ℕ, a ≤ p / 2 ∧ b ≤ p / 2 ∧ a ^ 2 + b ^ 2 ≡ x [MOD p] := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    x : Nat
    ⊢ Exists fun a => Exists fun b => And (LE.le a (HDiv.hDiv p 2)) (And (LE.le b  …
  -/
  simpa only [← Int.natCast_modEq_iff] using Nat.sq_add_sq_zmodEq p x
  /-
    🎉 no goals
  -/


theorem sq_add_sq (R : Type*) [CommRing R] [IsDomain R] (p : ℕ) [NeZero p] [CharP R p] (x : ℤ) :
    ∃ a b : ℕ, ((a : R) ^ 2 + (b : R) ^ 2) = x := by
  /-
    R : Type u_3
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    p : Nat
    inst✝¹ : NeZero p
    inst✝ : CharP R p
    x : Int
    ⊢ Exists fun a => Exists fun b => Eq (HAdd.hAdd (HPow.hPow (↑a) 2) (HPow.hPow  …
  -/
  haveI := char_is_prime_of_pos R p
  /-
    R : Type u_3
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    p : Nat
    inst✝¹ : NeZero p
    inst✝ : CharP R p
    x : Int
    this : Fact (Nat.Prime p)
    ⊢ Exists fun a => Exists fun b => Eq (HAdd.hAdd (HPow.hPow (↑a) 2) (HPow.hPow  …
  -/
  obtain ⟨a, b, hab⟩ := ZMod.sq_add_sq p x
  /-
    case intro.intro
    R : Type u_3
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    p : Nat
    inst✝¹ : NeZero p
    inst✝ : CharP R p
    x : Int
    this : Fact (Nat.Prime p)
    a b : ZMod p
    hab : Eq (HAdd.hAdd (HPow.hPow a 2) (HPow.hPow b 2)) ↑x
    ⊢ Exists fun a => Exists fun b => Eq (HAdd.hAdd (HPow.hPow (↑a) 2) (HPow.hPow  …
  -/
  refine ⟨a.val, b.val, ?_⟩
  /-
    case intro.intro
    R : Type u_3
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    p : Nat
    inst✝¹ : NeZero p
    inst✝ : CharP R p
    x : Int
    this : Fact (Nat.Prime p)
    a b : ZMod p
    hab : Eq (HAdd.hAdd (HPow.hPow a 2) (HPow.hPow b 2)) ↑x
    ⊢ Eq (HAdd.hAdd (HPow.hPow (↑a.val) 2) (HPow.hPow (↑b.val) 2)) ↑x
  -/
  simpa using congr_arg (ZMod.castHom dvd_rfl R) hab
  /-
    🎉 no goals
  -/


/-- The **Fermat-Euler totient theorem**. `Nat.ModEq.pow_totient` is an alternative statement
  of the same theorem. -/
@[simp]
theorem ZMod.pow_totient {n : ℕ} (x : (ZMod n)ˣ) : x ^ φ n = 1 := by
  /-
    n : Nat
    x : Units (ZMod n)
    ⊢ Eq (HPow.hPow x n.totient) 1
  -/
  cases n
    /-
      case zero
      x : Units (ZMod 0)
      ⊢ Eq (HPow.hPow x (Nat.totient 0)) 1
    -/
  · rw [Nat.totient_zero, pow_zero]
    /-
      🎉 no goals
    -/
    /-
      case succ
      n✝ : Nat
      x : Units (ZMod (HAdd.hAdd n✝ 1))
      ⊢ Eq (HPow.hPow x (HAdd.hAdd n✝ 1).totient) 1
    -/
  · rw [← card_units_eq_totient, pow_card_eq_one]
    /-
      🎉 no goals
    -/


/-- The **Fermat-Euler totient theorem**. `ZMod.pow_totient` is an alternative statement
  of the same theorem. -/
theorem Nat.ModEq.pow_totient {x n : ℕ} (h : Nat.Coprime x n) : x ^ φ n ≡ 1 [MOD n] := by
  /-
    x n : Nat
    h : x.Coprime n
    ⊢ n.ModEq (HPow.hPow x n.totient) 1
  -/
  rw [← ZMod.eq_iff_modEq_nat]
  /-
    x n : Nat
    h : x.Coprime n
    ⊢ Eq ↑(HPow.hPow x n.totient) ↑1
  -/
  let x' : Units (ZMod n) := ZMod.unitOfCoprime _ h
  /-
    x n : Nat
    h : x.Coprime n
    x' : Units (ZMod n) := ZMod.unitOfCoprime x h
    ⊢ Eq ↑(HPow.hPow x n.totient) ↑1
  -/
  have := ZMod.pow_totient x'
  /-
    x n : Nat
    h : x.Coprime n
    x' : Units (ZMod n) := ZMod.unitOfCoprime x h
    this : Eq (HPow.hPow x' n.totient) 1
    ⊢ Eq ↑(HPow.hPow x n.totient) ↑1
  -/
  apply_fun ((fun (x : Units (ZMod n)) => (x : ZMod n)) : Units (ZMod n) → ZMod n) at this
  simpa only [Nat.succ_eq_add_one, Nat.cast_pow, Units.val_one, Nat.cast_one,
    coe_unitOfCoprime, Units.val_pow_eq_pow_val]


/-- For each `n ≥ 0`, the unit group of `ZMod n` is finite. -/
instance instFiniteZModUnits : (n : ℕ) → Finite (ZMod n)ˣ
| 0     => Finite.of_fintype ℤˣ
| _ + 1 => inferInstance


theorem card_eq_pow_finrank [Fintype V] : Fintype.card V = q ^ Module.finrank K V := by
  /-
    K : Type u_1
    V : Type u_3
    inst✝⁴ : Fintype K
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Fintype V
    ⊢ Eq (Fintype.card V) (HPow.hPow (Fintype.card K) (Module.finrank K V))
  -/
  let b := IsNoetherian.finsetBasis K V
  /-
    K : Type u_1
    V : Type u_3
    inst✝⁴ : Fintype K
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Fintype V
    b : Basis (Subtype fun x => Membership.mem (IsNoetherian.finsetBasisIndex K V) …
    ⊢ Eq (Fintype.card V) (HPow.hPow (Fintype.card K) (Module.finrank K V))
  -/
  rw [Module.card_fintype b, ← Module.finrank_eq_card_basis b]
  /-
    🎉 no goals
  -/


/-- A variation on Fermat's little theorem. See `ZMod.pow_card_sub_one_eq_one` -/
@[simp]
theorem pow_card {p : ℕ} [Fact p.Prime] (x : ZMod p) : x ^ p = x := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    x : ZMod p
    ⊢ Eq (HPow.hPow x p) x
  -/
  have h := FiniteField.pow_card x; rwa [ZMod.card p] at h
                                    /-
                                      🎉 no goals
                                    -/


@[simp]
theorem pow_card_pow {n p : ℕ} [Fact p.Prime] (x : ZMod p) : x ^ p ^ n = x := by
  induction n with
  | zero => simp
  | succ n ih => simp [pow_succ, pow_mul, ih, pow_card]


@[simp]
theorem frobenius_zmod (p : ℕ) [Fact p.Prime] : frobenius (ZMod p) p = RingHom.id _ := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    ⊢ Eq (frobenius (ZMod p) p) (RingHom.id (ZMod p))
  -/
  ext a
  /-
    case a
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : ZMod p
    ⊢ Eq ((frobenius (ZMod p) p) a) ((RingHom.id (ZMod p)) a)
  -/
  rw [frobenius_def, ZMod.pow_card, RingHom.id_apply]
  /-
    🎉 no goals
  -/

-- Porting note: this was a `simp` lemma, but now the LHS simplify to `φ p`.

theorem card_units (p : ℕ) [Fact p.Prime] : Fintype.card (ZMod p)ˣ = p - 1 := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    ⊢ Eq (Fintype.card (Units (ZMod p))) (HSub.hSub p 1)
  -/
  rw [Fintype.card_units, card]
  /-
    🎉 no goals
  -/


/-- **Fermat's Little Theorem**: for every unit `a` of `ZMod p`, we have `a ^ (p - 1) = 1`. -/
theorem units_pow_card_sub_one_eq_one (p : ℕ) [Fact p.Prime] (a : (ZMod p)ˣ) : a ^ (p - 1) = 1 := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : Units (ZMod p)
    ⊢ Eq (HPow.hPow a (HSub.hSub p 1)) 1
  -/
  rw [← card_units p, pow_card_eq_one]
  /-
    🎉 no goals
  -/


/-- **Fermat's Little Theorem**: for all nonzero `a : ZMod p`, we have `a ^ (p - 1) = 1`. -/
theorem pow_card_sub_one_eq_one {p : ℕ} [Fact p.Prime] {a : ZMod p} (ha : a ≠ 0) :
    a ^ (p - 1) = 1 := by
    /-
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      a : ZMod p
      ha : Ne a 0
      ⊢ Eq (HPow.hPow a (HSub.hSub p 1)) 1
    -/
    have h := FiniteField.pow_card_sub_one_eq_one a ha
    /-
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      a : ZMod p
      ha : Ne a 0
      h : Eq (HPow.hPow a (HSub.hSub (Fintype.card (ZMod p)) 1)) 1
      ⊢ Eq (HPow.hPow a (HSub.hSub p 1)) 1
    -/
    rwa [ZMod.card p] at h
    /-
      🎉 no goals
    -/


lemma pow_card_sub_one {p : ℕ} [Fact p.Prime] (a : ZMod p) :
    a ^ (p - 1) = if a ≠ 0 then 1 else 0 := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : ZMod p
    ⊢ Eq (HPow.hPow a (HSub.hSub p 1)) (ite (Ne a 0) 1 0)
  -/
  split_ifs with ha
    /-
      case pos
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      a : ZMod p
      ha : Ne a 0
      ⊢ Eq (HPow.hPow a (HSub.hSub p 1)) 1
    -/
  · exact pow_card_sub_one_eq_one ha
    /-
      🎉 no goals
    -/
    /-
      case neg
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      a : ZMod p
      ha : Not (Ne a 0)
      ⊢ Eq (HPow.hPow a (HSub.hSub p 1)) 0
    -/
  · simp [of_not_not ha, (Fact.out : p.Prime).one_lt, tsub_eq_zero_iff_le]
    /-
      🎉 no goals
    -/


theorem orderOf_units_dvd_card_sub_one {p : ℕ} [Fact p.Prime] (u : (ZMod p)ˣ) : orderOf u ∣ p - 1 :=
  orderOf_dvd_of_pow_eq_one <| units_pow_card_sub_one_eq_one _ _


theorem orderOf_dvd_card_sub_one {p : ℕ} [Fact p.Prime] {a : ZMod p} (ha : a ≠ 0) :
    orderOf a ∣ p - 1 :=
  orderOf_dvd_of_pow_eq_one <| pow_card_sub_one_eq_one ha


theorem expand_card {p : ℕ} [Fact p.Prime] (f : Polynomial (ZMod p)) :
                                      /-
                                        p : Nat
                                        inst✝ : Fact (Nat.Prime p)
                                        f : Polynomial (ZMod p)
                                        ⊢ Eq ((Polynomial.expand (ZMod p) p) f) (HPow.hPow f p)
                                      -/
    expand (ZMod p) p f = f ^ p := by have h := FiniteField.expand_card f; rwa [ZMod.card p] at h
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


/-- **Fermat's Little Theorem**: for all `a : ℤ` coprime to `p`, we have
`a ^ (p - 1) ≡ 1 [ZMOD p]`. -/
theorem Int.ModEq.pow_card_sub_one_eq_one {p : ℕ} (hp : Nat.Prime p) {n : ℤ} (hpn : IsCoprime n p) :
    n ^ (p - 1) ≡ 1 [ZMOD p] := by
  /-
    p : Nat
    hp : Nat.Prime p
    n : Int
    hpn : IsCoprime n ↑p
    ⊢ (↑p).ModEq (HPow.hPow n (HSub.hSub p 1)) 1
  -/
  haveI : Fact p.Prime := ⟨hp⟩
  have : ¬(n : ZMod p) = 0 := by
    rw [CharP.intCast_eq_zero_iff _ p, ← (Nat.prime_iff_prime_int.mp hp).coprime_iff_not_dvd]
    · exact hpn.symm
  /-
    p : Nat
    hp : Nat.Prime p
    n : Int
    hpn : IsCoprime n ↑p
    this✝ : Fact (Nat.Prime p)
    this : Not (Eq (↑n) 0)
    ⊢ (↑p).ModEq (HPow.hPow n (HSub.hSub p 1)) 1
  -/
  simpa [← ZMod.intCast_eq_intCast_iff] using ZMod.pow_card_sub_one_eq_one this
  /-
    🎉 no goals
  -/


theorem pow_pow_modEq_one (p m a : ℕ) : (1 + p * a) ^ (p ^ m) ≡ 1 [MOD p ^ m] := by
  /-
    p m a : Nat
    ⊢ (HPow.hPow p m).ModEq (HPow.hPow (HAdd.hAdd 1 (HMul.hMul p a)) (HPow.hPow p  …
  -/
  induction' m with m hm
    /-
      case zero
      p a : Nat
      ⊢ (HPow.hPow p 0).ModEq (HPow.hPow (HAdd.hAdd 1 (HMul.hMul p a)) (HPow.hPow p  …
    -/
  · exact Nat.modEq_one
    /-
      🎉 no goals
    -/
    /-
      case succ
      p a m : Nat
      hm : (HPow.hPow p m).ModEq (HPow.hPow (HAdd.hAdd 1 (HMul.hMul p a)) (HPow.hPow …
      ⊢ (HPow.hPow p (HAdd.hAdd m 1)).ModEq (HPow.hPow (HAdd.hAdd 1 (HMul.hMul p a)) …
    -/
  · rw [Nat.ModEq.comm, add_comm, Nat.modEq_iff_dvd' (Nat.one_le_pow' _ _)] at hm
    /-
      case succ
      p a m : Nat
      hm : Dvd.dvd (HPow.hPow p m) (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul p a)  …
      ⊢ (HPow.hPow p (HAdd.hAdd m 1)).ModEq (HPow.hPow (HAdd.hAdd 1 (HMul.hMul p a)) …
    -/
    obtain ⟨d, hd⟩ := hm
    /-
      case succ.intro
      p a m d : Nat
      hd : Eq (HSub.hSub (HPow.hPow (HAdd.hAdd (HMul.hMul p a) 1) (HPow.hPow p m)) 1 …
      ⊢ (HPow.hPow p (HAdd.hAdd m 1)).ModEq (HPow.hPow (HAdd.hAdd 1 (HMul.hMul p a)) …
    -/
    rw [tsub_eq_iff_eq_add_of_le (Nat.one_le_pow' _ _), add_comm] at hd
    rw [pow_succ, pow_mul, hd, add_pow, Finset.sum_range_succ', pow_zero, one_mul, one_pow,
      one_mul, Nat.choose_zero_right, Nat.cast_one]
    /-
      case succ.intro
      p a m d : Nat
      hd : Eq (HPow.hPow (HAdd.hAdd 1 (HMul.hMul p a)) (HPow.hPow p m)) (HAdd.hAdd ( …
      ⊢ (HMul.hMul (HPow.hPow p m) p).ModEq (HAdd.hAdd ((Finset.range p).sum fun k = …
    -/
    refine Nat.ModEq.add_right 1 (Nat.modEq_zero_iff_dvd.mpr ?_)
    /-
      case succ.intro
      p a m d : Nat
      hd : Eq (HPow.hPow (HAdd.hAdd 1 (HMul.hMul p a)) (HPow.hPow p m)) (HAdd.hAdd ( …
      ⊢ Dvd.dvd (HMul.hMul (HPow.hPow p m) p) ((Finset.range p).sum fun k => HMul.hM …
    -/
    simp_rw [one_pow, mul_one, pow_succ', mul_assoc, ← Finset.mul_sum]
    /-
      case succ.intro
      p a m d : Nat
      hd : Eq (HPow.hPow (HAdd.hAdd 1 (HMul.hMul p a)) (HPow.hPow p m)) (HAdd.hAdd ( …
      ⊢ Dvd.dvd (HMul.hMul (HPow.hPow p m) p) (HMul.hMul (HPow.hPow p m) (HMul.hMul  …
    -/
    refine mul_dvd_mul_left (p ^ m) (dvd_mul_of_dvd_right (Finset.dvd_sum fun k hk ↦ ?_) d)
    /-
      case succ.intro
      p a m d : Nat
      hd : Eq (HPow.hPow (HAdd.hAdd 1 (HMul.hMul p a)) (HPow.hPow p m)) (HAdd.hAdd ( …
      k : Nat
      hk : Membership.mem (Finset.range p) k
      ⊢ Dvd.dvd p (HMul.hMul (HPow.hPow (HMul.hMul (HPow.hPow p m) d) k) ↑(p.choose  …
    -/
    cases m
      /-
        case succ.intro.zero
        p a d k : Nat
        hk : Membership.mem (Finset.range p) k
        hd : Eq (HPow.hPow (HAdd.hAdd 1 (HMul.hMul p a)) (HPow.hPow p 0)) (HAdd.hAdd ( …
        ⊢ Dvd.dvd p (HMul.hMul (HPow.hPow (HMul.hMul (HPow.hPow p 0) d) k) ↑(p.choose  …
      -/
    · rw [pow_zero, pow_one, one_mul, add_comm, add_left_inj] at hd
      /-
        case succ.intro.zero
        p a d k : Nat
        hk : Membership.mem (Finset.range p) k
        hd : Eq (HMul.hMul p a) d
        ⊢ Dvd.dvd p (HMul.hMul (HPow.hPow (HMul.hMul (HPow.hPow p 0) d) k) ↑(p.choose  …
      -/
                  /-
                    🎉 no goals
                  -/
      cases k <;> simp [← hd, mul_assoc, pow_succ']
                  /-
                    🎉 no goals
                  -/
      /-
        case succ.intro.succ
        p a d k : Nat
        hk : Membership.mem (Finset.range p) k
        n✝ : Nat
        hd : Eq (HPow.hPow (HAdd.hAdd 1 (HMul.hMul p a)) (HPow.hPow p (HAdd.hAdd n✝ 1) …
        ⊢ Dvd.dvd p (HMul.hMul (HPow.hPow (HMul.hMul (HPow.hPow p (HAdd.hAdd n✝ 1)) d) …
      -/
                  /-
                    🎉 no goals
                  -/
    · cases k <;> simp [mul_assoc, pow_succ']
                  /-
                    🎉 no goals
                  -/


theorem ZMod.eq_one_or_isUnit_sub_one {n p k : ℕ} [Fact p.Prime] (hn : n = p ^ k) (a : ZMod n)
    (ha : (orderOf a).Coprime n) : a = 1 ∨ IsUnit (a - 1) := by
  /-
    n p k : Nat
    inst✝ : Fact (Nat.Prime p)
    hn : Eq n (HPow.hPow p k)
    a : ZMod n
    ha : (orderOf a).Coprime n
    ⊢ Or (Eq a 1) (IsUnit (HSub.hSub a 1))
  -/
  rcases eq_or_ne n 0 with rfl | hn0
    /-
      case inl
      p k : Nat
      inst✝ : Fact (Nat.Prime p)
      hn : Eq 0 (HPow.hPow p k)
      a : ZMod 0
      ha : (orderOf a).Coprime 0
      ⊢ Or (Eq a 1) (IsUnit (HSub.hSub a 1))
    -/
  · exact Or.inl (orderOf_eq_one_iff.mp ((orderOf a).coprime_zero_right.mp ha))
    /-
      🎉 no goals
    -/
  /-
    case inr
    n p k : Nat
    inst✝ : Fact (Nat.Prime p)
    hn : Eq n (HPow.hPow p k)
    a : ZMod n
    ha : (orderOf a).Coprime n
    hn0 : Ne n 0
    ⊢ Or (Eq a 1) (IsUnit (HSub.hSub a 1))
  -/
  rcases eq_or_ne a 0 with rfl | ha0
    /-
      case inr.inl
      n p k : Nat
      inst✝ : Fact (Nat.Prime p)
      hn : Eq n (HPow.hPow p k)
      hn0 : Ne n 0
      ha : (orderOf 0).Coprime n
      ⊢ Or (Eq 0 1) (IsUnit (HSub.hSub 0 1))
    -/
  · exact Or.inr (zero_sub (1 : ZMod n) ▸ isUnit_neg_one)
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    n p k : Nat
    inst✝ : Fact (Nat.Prime p)
    hn : Eq n (HPow.hPow p k)
    a : ZMod n
    ha : (orderOf a).Coprime n
    hn0 : Ne n 0
    ha0 : Ne a 0
    ⊢ Or (Eq a 1) (IsUnit (HSub.hSub a 1))
  -/
  have : NeZero n := ⟨hn0⟩
  /-
    case inr.inr
    n p k : Nat
    inst✝ : Fact (Nat.Prime p)
    hn : Eq n (HPow.hPow p k)
    a : ZMod n
    ha : (orderOf a).Coprime n
    hn0 : Ne n 0
    ha0 : Ne a 0
    this : NeZero n
    ⊢ Or (Eq a 1) (IsUnit (HSub.hSub a 1))
  -/
  obtain ⟨a, rfl⟩ := ZMod.natCast_zmod_surjective a
  /-
    case inr.inr.intro
    n p k : Nat
    inst✝ : Fact (Nat.Prime p)
    hn : Eq n (HPow.hPow p k)
    hn0 : Ne n 0
    this : NeZero n
    a : Nat
    ha : (orderOf ↑a).Coprime n
    ha0 : Ne (↑a) 0
    ⊢ Or (Eq (↑a) 1) (IsUnit (HSub.hSub (↑a) 1))
  -/
  rw [← orderOf_eq_one_iff, or_iff_not_imp_right]
  /-
    case inr.inr.intro
    n p k : Nat
    inst✝ : Fact (Nat.Prime p)
    hn : Eq n (HPow.hPow p k)
    hn0 : Ne n 0
    this : NeZero n
    a : Nat
    ha : (orderOf ↑a).Coprime n
    ha0 : Ne (↑a) 0
    ⊢ Not (IsUnit (HSub.hSub (↑a) 1)) → Eq (orderOf ↑a) 1
  -/
  refine fun h ↦ ha.eq_one_of_dvd ?_
  /-
    case inr.inr.intro
    n p k : Nat
    inst✝ : Fact (Nat.Prime p)
    hn : Eq n (HPow.hPow p k)
    hn0 : Ne n 0
    this : NeZero n
    a : Nat
    ha : (orderOf ↑a).Coprime n
    ha0 : Ne (↑a) 0
    h : Not (IsUnit (HSub.hSub (↑a) 1))
    ⊢ Dvd.dvd (orderOf ↑a) n
  -/
  rw [orderOf_dvd_iff_pow_eq_one, ← Nat.cast_pow, ← Nat.cast_one, ZMod.eq_iff_modEq_nat, hn]
  replace ha0 : 1 ≤ a := by
    contrapose! ha0
    rw [Nat.lt_one_iff.mp ha0, Nat.cast_zero]
  /-
    case inr.inr.intro
    n p k : Nat
    inst✝ : Fact (Nat.Prime p)
    hn : Eq n (HPow.hPow p k)
    hn0 : Ne n 0
    this : NeZero n
    a : Nat
    ha : (orderOf ↑a).Coprime n
    h : Not (IsUnit (HSub.hSub (↑a) 1))
    ha0 : LE.le 1 a
    ⊢ (HPow.hPow p k).ModEq (HPow.hPow a (HPow.hPow p k)) 1
  -/
  rw [← Nat.cast_one, ← Nat.cast_sub ha0, ZMod.isUnit_iff_coprime, hn] at h
  /-
    case inr.inr.intro
    n p k : Nat
    inst✝ : Fact (Nat.Prime p)
    hn : Eq n (HPow.hPow p k)
    hn0 : Ne n 0
    this : NeZero n
    a : Nat
    ha : (orderOf ↑a).Coprime n
    h : Not ((HSub.hSub a 1).Coprime (HPow.hPow p k))
    ha0 : LE.le 1 a
    ⊢ (HPow.hPow p k).ModEq (HPow.hPow a (HPow.hPow p k)) 1
  -/
  obtain ⟨b, hb⟩ := not_imp_comm.mp (Nat.Prime.coprime_pow_of_not_dvd Fact.out) h
  /-
    case inr.inr.intro.intro
    n p k : Nat
    inst✝ : Fact (Nat.Prime p)
    hn : Eq n (HPow.hPow p k)
    hn0 : Ne n 0
    this : NeZero n
    a : Nat
    ha : (orderOf ↑a).Coprime n
    h : Not ((HSub.hSub a 1).Coprime (HPow.hPow p k))
    ha0 : LE.le 1 a
    b : Nat
    hb : Eq (HSub.hSub a 1) (HMul.hMul p b)
    ⊢ (HPow.hPow p k).ModEq (HPow.hPow a (HPow.hPow p k)) 1
  -/
  rw [tsub_eq_iff_eq_add_of_le ha0, add_comm] at hb
  /-
    case inr.inr.intro.intro
    n p k : Nat
    inst✝ : Fact (Nat.Prime p)
    hn : Eq n (HPow.hPow p k)
    hn0 : Ne n 0
    this : NeZero n
    a : Nat
    ha : (orderOf ↑a).Coprime n
    h : Not ((HSub.hSub a 1).Coprime (HPow.hPow p k))
    ha0 : LE.le 1 a
    b : Nat
    hb : Eq a (HAdd.hAdd 1 (HMul.hMul p b))
    ⊢ (HPow.hPow p k).ModEq (HPow.hPow a (HPow.hPow p k)) 1
  -/
  exact hb ▸ pow_pow_modEq_one p k b
  /-
    🎉 no goals
  -/


/-- In a finite field of characteristic `2`, all elements are squares. -/
theorem isSquare_of_char_two (hF : ringChar F = 2) (a : F) : IsSquare a :=
  haveI hF' : CharP F 2 := ringChar.of_eq hF
  isSquare_of_charTwo' a


/-- In a finite field of odd characteristic, not every element is a square. -/
theorem exists_nonsquare (hF : ringChar F ≠ 2) : ∃ a : F, ¬IsSquare a := by
  -- Idea: the squaring map on `F` is not injective, hence not surjective
  have h : ¬Function.Injective fun x : F ↦ x * x := fun h ↦
    h.ne (Ring.neg_one_ne_one_of_char_ne_two hF) <| by simp
  /-
    F : Type u_3
    inst✝¹ : Field F
    inst✝ : Finite F
    hF : Ne (ringChar F) 2
    h : Not (Function.Injective fun x => HMul.hMul x x)
    ⊢ Exists fun a => Not (IsSquare a)
  -/
  simpa [Finite.injective_iff_surjective, Function.Surjective, IsSquare, eq_comm] using h
  /-
    🎉 no goals
  -/


/-- The finite field `F` has even cardinality iff it has characteristic `2`. -/
theorem even_card_iff_char_two : ringChar F = 2 ↔ Fintype.card F % 2 = 0 := by
  /-
    F : Type u_3
    inst✝¹ : Field F
    inst✝ : Fintype F
    ⊢ Iff (Eq (ringChar F) 2) (Eq (HMod.hMod (Fintype.card F) 2) 0)
  -/
  rcases FiniteField.card F (ringChar F) with ⟨n, hp, h⟩
  /-
    case intro.intro
    F : Type u_3
    inst✝¹ : Field F
    inst✝ : Fintype F
    n : PNat
    hp : Nat.Prime (ringChar F)
    h : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    ⊢ Iff (Eq (ringChar F) 2) (Eq (HMod.hMod (Fintype.card F) 2) 0)
  -/
  rw [h, ← Nat.even_iff, Nat.even_pow, hp.even_iff]
  /-
    case intro.intro
    F : Type u_3
    inst✝¹ : Field F
    inst✝ : Fintype F
    n : PNat
    hp : Nat.Prime (ringChar F)
    h : Eq (Fintype.card F) (HPow.hPow (ringChar F) ↑n)
    ⊢ Iff (Eq (ringChar F) 2) (And (Eq (ringChar F) 2) (Ne (↑n) 0))
  -/
  simp
  /-
    🎉 no goals
  -/


theorem even_card_of_char_two (hF : ringChar F = 2) : Fintype.card F % 2 = 0 :=
  even_card_iff_char_two.mp hF


theorem odd_card_of_char_ne_two (hF : ringChar F ≠ 2) : Fintype.card F % 2 = 1 :=
  Nat.mod_two_ne_zero.mp (mt even_card_iff_char_two.mpr hF)


/-- If `F` has odd characteristic, then for nonzero `a : F`, we have that `a ^ (#F / 2) = ±1`. -/
theorem pow_dichotomy (hF : ringChar F ≠ 2) {a : F} (ha : a ≠ 0) :
    a ^ (Fintype.card F / 2) = 1 ∨ a ^ (Fintype.card F / 2) = -1 := by
  /-
    F : Type u_3
    inst✝¹ : Field F
    inst✝ : Fintype F
    hF : Ne (ringChar F) 2
    a : F
    ha : Ne a 0
    ⊢ Or (Eq (HPow.hPow a (HDiv.hDiv (Fintype.card F) 2)) 1) (Eq (HPow.hPow a (HDi …
  -/
  have h₁ := FiniteField.pow_card_sub_one_eq_one a ha
  rw [← Nat.two_mul_odd_div_two (FiniteField.odd_card_of_char_ne_two hF), mul_comm, pow_mul,
    pow_two] at h₁
  /-
    F : Type u_3
    inst✝¹ : Field F
    inst✝ : Fintype F
    hF : Ne (ringChar F) 2
    a : F
    ha : Ne a 0
    h₁ : Eq (HMul.hMul (HPow.hPow a (HDiv.hDiv (Fintype.card F) 2)) (HPow.hPow a ( …
    ⊢ Or (Eq (HPow.hPow a (HDiv.hDiv (Fintype.card F) 2)) 1) (Eq (HPow.hPow a (HDi …
  -/
  exact mul_self_eq_one_iff.mp h₁
  /-
    🎉 no goals
  -/


/-- A unit `a` of a finite field `F` of odd characteristic is a square
if and only if `a ^ (#F / 2) = 1`. -/
theorem unit_isSquare_iff (hF : ringChar F ≠ 2) (a : Fˣ) :
    IsSquare a ↔ a ^ (Fintype.card F / 2) = 1 := by
  classical
    obtain ⟨g, hg⟩ := IsCyclic.exists_generator (α := Fˣ)
    obtain ⟨n, hn⟩ : a ∈ Submonoid.powers g := by rw [mem_powers_iff_mem_zpowers]; apply hg
    have hodd := Nat.two_mul_odd_div_two (FiniteField.odd_card_of_char_ne_two hF)
    constructor
    · rintro ⟨y, rfl⟩
      rw [← pow_two, ← pow_mul, hodd]
      apply_fun Units.val using Units.ext
      push_cast
      exact FiniteField.pow_card_sub_one_eq_one (y : F) (Units.ne_zero y)
    · subst a; intro h
      rw [← Nat.card_eq_fintype_card] at hodd h
      have key : 2 * (Nat.card F / 2) ∣ n * (Nat.card F / 2) := by
        rw [← pow_mul] at h
        rw [hodd, ← Nat.card_units, ← orderOf_eq_card_of_forall_mem_zpowers hg]
        apply orderOf_dvd_of_pow_eq_one h
      have : 0 < Nat.card F / 2 := Nat.div_pos Finite.one_lt_card (by norm_num)
      obtain ⟨m, rfl⟩ := Nat.dvd_of_mul_dvd_mul_right this key
      refine ⟨g ^ m, ?_⟩
      dsimp
      rw [mul_comm, pow_mul, pow_two]


/-- A non-zero `a : F` is a square if and only if `a ^ (#F / 2) = 1`. -/
theorem isSquare_iff (hF : ringChar F ≠ 2) {a : F} (ha : a ≠ 0) :
    IsSquare a ↔ a ^ (Fintype.card F / 2) = 1 := by
  apply
    (iff_congr _ (by simp [Units.ext_iff])).mp (FiniteField.unit_isSquare_iff hF (Units.mk0 a ha))
  /-
    F : Type u_3
    inst✝¹ : Field F
    inst✝ : Fintype F
    hF : Ne (ringChar F) 2
    a : F
    ha : Ne a 0
    ⊢ Iff (IsSquare (Units.mk0 a ha)) (IsSquare a)
  -/
  simp only [IsSquare, Units.ext_iff, Units.val_mk0, Units.val_mul]
  /-
    F : Type u_3
    inst✝¹ : Field F
    inst✝ : Fintype F
    hF : Ne (ringChar F) 2
    a : F
    ha : Ne a 0
    ⊢ Iff (Exists fun r => Eq a (HMul.hMul ↑r ↑r)) (Exists fun r => Eq a (HMul.hMu …
  -/
  constructor
    /-
      case mp
      F : Type u_3
      inst✝¹ : Field F
      inst✝ : Fintype F
      hF : Ne (ringChar F) 2
      a : F
      ha : Ne a 0
      ⊢ (Exists fun r => Eq a (HMul.hMul ↑r ↑r)) → Exists fun r => Eq a (HMul.hMul r …
    -/
  · rintro ⟨y, hy⟩; exact ⟨y, hy⟩
                    /-
                      🎉 no goals
                    -/
    /-
      case mpr
      F : Type u_3
      inst✝¹ : Field F
      inst✝ : Fintype F
      hF : Ne (ringChar F) 2
      a : F
      ha : Ne a 0
      ⊢ (Exists fun r => Eq a (HMul.hMul r r)) → Exists fun r => Eq a (HMul.hMul ↑r  …
    -/
  · rintro ⟨y, rfl⟩
    /-
      case mpr.intro
      F : Type u_3
      inst✝¹ : Field F
      inst✝ : Fintype F
      hF : Ne (ringChar F) 2
      y : F
      ha : Ne (HMul.hMul y y) 0
      ⊢ Exists fun r => Eq (HMul.hMul y y) (HMul.hMul ↑r ↑r)
    -/
    have hy : y ≠ 0 := by rintro rfl; simp at ha
    /-
      case mpr.intro
      F : Type u_3
      inst✝¹ : Field F
      inst✝ : Fintype F
      hF : Ne (ringChar F) 2
      y : F
      ha : Ne (HMul.hMul y y) 0
      hy : Ne y 0
      ⊢ Exists fun r => Eq (HMul.hMul y y) (HMul.hMul ↑r ↑r)
    -/
    refine ⟨Units.mk0 y hy, ?_⟩; simp
                                 /-
                                   🎉 no goals
                                 -/


