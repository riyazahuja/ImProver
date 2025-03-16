/-- The Jacobi symbol of `a` and `b` -/
def jacobiSym (a : ℤ) (b : ℕ) : ℤ :=
  (b.primeFactorsList.pmap (fun p pp => @legendreSym p ⟨pp⟩ a) fun _ pf =>
    prime_of_mem_primeFactorsList pf).prod

-- Notation for the Jacobi symbol.

@[inherit_doc]
scoped[NumberTheorySymbols] notation "J(" a " | " b ")" => jacobiSym a b

-- Porting note: Without the following line, Lean expected `|` on several lines, e.g. line 102.

/-- The symbol `J(a | 0)` has the value `1`. -/
@[simp]
theorem zero_right (a : ℤ) : J(a | 0) = 1 := by
  /-
    a : Int
    ⊢ Eq (jacobiSym a 0) 1
  -/
  simp only [jacobiSym, primeFactorsList_zero, List.prod_nil, List.pmap]
  /-
    🎉 no goals
  -/


/-- The symbol `J(a | 1)` has the value `1`. -/
@[simp]
theorem one_right (a : ℤ) : J(a | 1) = 1 := by
  /-
    a : Int
    ⊢ Eq (jacobiSym a 1) 1
  -/
  simp only [jacobiSym, primeFactorsList_one, List.prod_nil, List.pmap]
  /-
    🎉 no goals
  -/


/-- The Legendre symbol `legendreSym p a` with an integer `a` and a prime number `p`
is the same as the Jacobi symbol `J(a | p)`. -/
theorem legendreSym.to_jacobiSym (p : ℕ) [fp : Fact p.Prime] (a : ℤ) :
    legendreSym p a = J(a | p) := by
  simp only [jacobiSym, primeFactorsList_prime fp.1, List.prod_cons, List.prod_nil, mul_one,
    List.pmap]


/-- The Jacobi symbol is multiplicative in its second argument. -/
theorem mul_right' (a : ℤ) {b₁ b₂ : ℕ} (hb₁ : b₁ ≠ 0) (hb₂ : b₂ ≠ 0) :
    J(a | b₁ * b₂) = J(a | b₁) * J(a | b₂) := by
  rw [jacobiSym, ((perm_primeFactorsList_mul hb₁ hb₂).pmap _).prod_eq, List.pmap_append,
    List.prod_append]
  case h => exact fun p hp =>
    (List.mem_append.mp hp).elim prime_of_mem_primeFactorsList prime_of_mem_primeFactorsList
  /-
    a : Int
    b₁ b₂ : Nat
    hb₁ : Ne b₁ 0
    hb₂ : Ne b₂ 0
    ⊢ Eq (HMul.hMul (List.pmap (fun p pp => legendreSym p a) b₁.primeFactorsList ⋯ …
  -/
  case _ => rfl
  /-
    🎉 no goals
  -/


/-- The Jacobi symbol is multiplicative in its second argument. -/
theorem mul_right (a : ℤ) (b₁ b₂ : ℕ) [NeZero b₁] [NeZero b₂] :
    J(a | b₁ * b₂) = J(a | b₁) * J(a | b₂) :=
  mul_right' a (NeZero.ne b₁) (NeZero.ne b₂)


/-- The Jacobi symbol takes only the values `0`, `1` and `-1`. -/
theorem trichotomy (a : ℤ) (b : ℕ) : J(a | b) = 0 ∨ J(a | b) = 1 ∨ J(a | b) = -1 :=
  ((@SignType.castHom ℤ _ _).toMonoidHom.mrange.copy {0, 1, -1} <| by
    /-
      a : Int
      b : Nat
      ⊢ Eq (Insert.insert 0 (Insert.insert 1 (Singleton.singleton (-1)))) ↑(MonoidHo …
    -/
    rw [Set.pair_comm]
    /-
      a : Int
      b : Nat
      ⊢ Eq (Insert.insert 0 (Insert.insert (-1) (Singleton.singleton 1))) ↑(MonoidHo …
    -/
    exact (SignType.range_eq SignType.castHom).symm).list_prod_mem
    /-
      🎉 no goals
    -/
      (by
        /-
          a : Int
          b : Nat
          ⊢ ∀ (x : Int), Membership.mem (List.pmap (fun p pp => legendreSym p a) b.prime …
        -/
        intro _ ha'
        /-
          a : Int
          b : Nat
          x✝ : Int
          ha' : Membership.mem (List.pmap (fun p pp => legendreSym p a) b.primeFactorsLi …
          ⊢ Membership.mem ((MonoidHom.mrange ↑SignType.castHom).copy (Insert.insert 0 ( …
        -/
        rcases List.mem_pmap.mp ha' with ⟨p, hp, rfl⟩
        /-
          case intro.intro
          a : Int
          b p : Nat
          hp : Membership.mem b.primeFactorsList p
          ha' : Membership.mem (List.pmap (fun p pp => legendreSym p a) b.primeFactorsLi …
          ⊢ Membership.mem ((MonoidHom.mrange ↑SignType.castHom).copy (Insert.insert 0 ( …
        -/
        haveI : Fact p.Prime := ⟨prime_of_mem_primeFactorsList hp⟩
        /-
          case intro.intro
          a : Int
          b p : Nat
          hp : Membership.mem b.primeFactorsList p
          ha' : Membership.mem (List.pmap (fun p pp => legendreSym p a) b.primeFactorsLi …
          this : Fact (Nat.Prime p)
          ⊢ Membership.mem ((MonoidHom.mrange ↑SignType.castHom).copy (Insert.insert 0 ( …
        -/
        exact quadraticChar_isQuadratic (ZMod p) a)
        /-
          🎉 no goals
        -/


/-- The symbol `J(1 | b)` has the value `1`. -/
@[simp]
theorem one_left (b : ℕ) : J(1 | b) = 1 :=
  List.prod_eq_one fun z hz => by
    /-
      b : Nat
      z : Int
      hz : Membership.mem (List.pmap (fun p pp => legendreSym p 1) b.primeFactorsLis …
      ⊢ Eq z 1
    -/
    let ⟨p, hp, he⟩ := List.mem_pmap.1 hz
    -- Porting note: The line 150 was added because Lean does not synthesize the instance
    -- `[Fact (Nat.Prime p)]` automatically (it is needed for `legendreSym.at_one`)
    /-
      b : Nat
      z : Int
      hz : Membership.mem (List.pmap (fun p pp => legendreSym p 1) b.primeFactorsLis …
      p : Nat
      hp : Membership.mem b.primeFactorsList p
      he : Eq (legendreSym p 1) z
      ⊢ Eq z 1
    -/
    letI : Fact p.Prime := ⟨prime_of_mem_primeFactorsList hp⟩
    /-
      b : Nat
      z : Int
      hz : Membership.mem (List.pmap (fun p pp => legendreSym p 1) b.primeFactorsLis …
      p : Nat
      hp : Membership.mem b.primeFactorsList p
      he : Eq (legendreSym p 1) z
      this : Fact (Nat.Prime p) := { out := Nat.prime_of_mem_primeFactorsList hp }
      ⊢ Eq z 1
    -/
    rw [← he, legendreSym.at_one]
    /-
      🎉 no goals
    -/


/-- The Jacobi symbol is multiplicative in its first argument. -/
theorem mul_left (a₁ a₂ : ℤ) (b : ℕ) : J(a₁ * a₂ | b) = J(a₁ | b) * J(a₂ | b) := by
  /-
    a₁ a₂ : Int
    b : Nat
    ⊢ Eq (jacobiSym (HMul.hMul a₁ a₂) b) (HMul.hMul (jacobiSym a₁ b) (jacobiSym a₂ …
  -/
  simp_rw [jacobiSym, List.pmap_eq_map_attach, legendreSym.mul _ _ _]
  exact List.prod_map_mul (α := ℤ) (l := (primeFactorsList b).attach)
    (f := fun x ↦ @legendreSym x {out := prime_of_mem_primeFactorsList x.2} a₁)
    (g := fun x ↦ @legendreSym x {out := prime_of_mem_primeFactorsList x.2} a₂)


/-- The symbol `J(a | b)` vanishes iff `a` and `b` are not coprime (assuming `b ≠ 0`). -/
theorem eq_zero_iff_not_coprime {a : ℤ} {b : ℕ} [NeZero b] : J(a | b) = 0 ↔ a.gcd b ≠ 1 :=
  List.prod_eq_zero_iff.trans
    (by
      /-
        a : Int
        b : Nat
        inst✝ : NeZero b
        ⊢ Iff (Membership.mem (List.pmap (fun p pp => legendreSym p a) b.primeFactorsL …
      -/
      rw [List.mem_pmap, Int.gcd_eq_natAbs, Ne, Prime.not_coprime_iff_dvd]
      simp_rw [legendreSym.eq_zero_iff _ _, intCast_zmod_eq_zero_iff_dvd,
        mem_primeFactorsList (NeZero.ne b), ← Int.natCast_dvd, Int.natCast_dvd_natCast, exists_prop,
        and_assoc, _root_.and_comm])


/-- The symbol `J(a | b)` is nonzero when `a` and `b` are coprime. -/
protected theorem ne_zero {a : ℤ} {b : ℕ} (h : a.gcd b = 1) : J(a | b) ≠ 0 := by
  /-
    a : Int
    b : Nat
    h : Eq (a.gcd ↑b) 1
    ⊢ Ne (jacobiSym a b) 0
  -/
  cases' eq_zero_or_neZero b with hb
    /-
      case inl
      a : Int
      b : Nat
      h : Eq (a.gcd ↑b) 1
      hb : Eq b 0
      ⊢ Ne (jacobiSym a b) 0
    -/
  · rw [hb, zero_right]
    /-
      case inl
      a : Int
      b : Nat
      h : Eq (a.gcd ↑b) 1
      hb : Eq b 0
      ⊢ Ne 1 0
    -/
    exact one_ne_zero
    /-
      🎉 no goals
    -/
    /-
      case inr
      a : Int
      b : Nat
      h : Eq (a.gcd ↑b) 1
      h✝ : NeZero b
      ⊢ Ne (jacobiSym a b) 0
    -/
  · contrapose! h; exact eq_zero_iff_not_coprime.1 h
                   /-
                     🎉 no goals
                   -/


/-- The symbol `J(a | b)` vanishes if and only if `b ≠ 0` and `a` and `b` are not coprime. -/
theorem eq_zero_iff {a : ℤ} {b : ℕ} : J(a | b) = 0 ↔ b ≠ 0 ∧ a.gcd b ≠ 1 :=
  ⟨fun h => by
    /-
      a : Int
      b : Nat
      h : Eq (jacobiSym a b) 0
      ⊢ And (Ne b 0) (Ne (a.gcd ↑b) 1)
    -/
    rcases eq_or_ne b 0 with hb | hb
      /-
        case inl
        a : Int
        b : Nat
        h : Eq (jacobiSym a b) 0
        hb : Eq b 0
        ⊢ And (Ne b 0) (Ne (a.gcd ↑b) 1)
      -/
    · rw [hb, zero_right] at h; cases h
                                /-
                                  🎉 no goals
                                -/
    /-
      case inr
      a : Int
      b : Nat
      h : Eq (jacobiSym a b) 0
      hb : Ne b 0
      ⊢ And (Ne b 0) (Ne (a.gcd ↑b) 1)
    -/
    exact ⟨hb, mt jacobiSym.ne_zero <| Classical.not_not.2 h⟩, fun ⟨hb, h⟩ => by
    /-
      🎉 no goals
    -/
    /-
      a : Int
      b : Nat
      x✝ : And (Ne b 0) (Ne (a.gcd ↑b) 1)
      hb : Ne b 0
      h : Ne (a.gcd ↑b) 1
      ⊢ Eq (jacobiSym a b) 0
    -/
    rw [← neZero_iff] at hb; exact eq_zero_iff_not_coprime.2 h⟩
                             /-
                               🎉 no goals
                             -/


/-- The symbol `J(0 | b)` vanishes when `b > 1`. -/
theorem zero_left {b : ℕ} (hb : 1 < b) : J(0 | b) = 0 :=
  (@eq_zero_iff_not_coprime 0 b ⟨ne_zero_of_lt hb⟩).mpr <| by
    /-
      b : Nat
      hb : LT.lt 1 b
      ⊢ Ne (Int.gcd 0 ↑b) 1
    -/
    rw [Int.gcd_zero_left, Int.natAbs_ofNat]; exact hb.ne'
                                              /-
                                                🎉 no goals
                                              -/


/-- The symbol `J(a | b)` takes the value `1` or `-1` if `a` and `b` are coprime. -/
theorem eq_one_or_neg_one {a : ℤ} {b : ℕ} (h : a.gcd b = 1) : J(a | b) = 1 ∨ J(a | b) = -1 :=
  (trichotomy a b).resolve_left <| jacobiSym.ne_zero h


/-- We have that `J(a^e | b) = J(a | b)^e`. -/
theorem pow_left (a : ℤ) (e b : ℕ) : J(a ^ e | b) = J(a | b) ^ e :=
                  /-
                    a : Int
                    e b : Nat
                    ⊢ Eq (jacobiSym (HPow.hPow a Nat.zero) b) (HPow.hPow (jacobiSym a b) Nat.zero)
                  -/
  Nat.recOn e (by rw [_root_.pow_zero, _root_.pow_zero, one_left]) fun _ ih => by
                  /-
                    🎉 no goals
                  -/
    /-
      a : Int
      e b x✝ : Nat
      ih : Eq (jacobiSym (HPow.hPow a x✝) b) (HPow.hPow (jacobiSym a b) x✝)
      ⊢ Eq (jacobiSym (HPow.hPow a x✝.succ) b) (HPow.hPow (jacobiSym a b) x✝.succ)
    -/
    rw [_root_.pow_succ, _root_.pow_succ, mul_left, ih]
    /-
      🎉 no goals
    -/


/-- We have that `J(a | b^e) = J(a | b)^e`. -/
theorem pow_right (a : ℤ) (b e : ℕ) : J(a | b ^ e) = J(a | b) ^ e := by
  /-
    a : Int
    b e : Nat
    ⊢ Eq (jacobiSym a (HPow.hPow b e)) (HPow.hPow (jacobiSym a b) e)
  -/
  induction' e with e ih
    /-
      case zero
      a : Int
      b : Nat
      ⊢ Eq (jacobiSym a (HPow.hPow b 0)) (HPow.hPow (jacobiSym a b) 0)
    -/
  · rw [Nat.pow_zero, _root_.pow_zero, one_right]
    /-
      🎉 no goals
    -/
    /-
      case succ
      a : Int
      b e : Nat
      ih : Eq (jacobiSym a (HPow.hPow b e)) (HPow.hPow (jacobiSym a b) e)
      ⊢ Eq (jacobiSym a (HPow.hPow b (HAdd.hAdd e 1))) (HPow.hPow (jacobiSym a b) (H …
    -/
  · cases' eq_zero_or_neZero b with hb
      /-
        case succ.inl
        a : Int
        b e : Nat
        ih : Eq (jacobiSym a (HPow.hPow b e)) (HPow.hPow (jacobiSym a b) e)
        hb : Eq b 0
        ⊢ Eq (jacobiSym a (HPow.hPow b (HAdd.hAdd e 1))) (HPow.hPow (jacobiSym a b) (H …
      -/
    · rw [hb, zero_pow e.succ_ne_zero, zero_right, one_pow]
      /-
        🎉 no goals
      -/
      /-
        case succ.inr
        a : Int
        b e : Nat
        ih : Eq (jacobiSym a (HPow.hPow b e)) (HPow.hPow (jacobiSym a b) e)
        h✝ : NeZero b
        ⊢ Eq (jacobiSym a (HPow.hPow b (HAdd.hAdd e 1))) (HPow.hPow (jacobiSym a b) (H …
      -/
    · rw [_root_.pow_succ, _root_.pow_succ, mul_right, ih]
      /-
        🎉 no goals
      -/


/-- The square of `J(a | b)` is `1` when `a` and `b` are coprime. -/
theorem sq_one {a : ℤ} {b : ℕ} (h : a.gcd b = 1) : J(a | b) ^ 2 = 1 := by
  /-
    a : Int
    b : Nat
    h : Eq (a.gcd ↑b) 1
    ⊢ Eq (HPow.hPow (jacobiSym a b) 2) 1
  -/
                                                        /-
                                                          🎉 no goals
                                                        -/
  cases' eq_one_or_neg_one h with h₁ h₁ <;> rw [h₁] <;> rfl
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- The symbol `J(a^2 | b)` is `1` when `a` and `b` are coprime. -/
                                                                           /-
                                                                             a : Int
                                                                             b : Nat
                                                                             h : Eq (a.gcd ↑b) 1
                                                                             ⊢ Eq (jacobiSym (HPow.hPow a 2) b) 1
                                                                           -/
theorem sq_one' {a : ℤ} {b : ℕ} (h : a.gcd b = 1) : J(a ^ 2 | b) = 1 := by rw [pow_left, sq_one h]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


/-- The symbol `J(a | b)` depends only on `a` mod `b`. -/
theorem mod_left (a : ℤ) (b : ℕ) : J(a | b) = J(a % b | b) :=
  congr_arg List.prod <|
    List.pmap_congr_left _
      (by
        -- Porting note: Lean does not synthesize the instance [Fact (Nat.Prime p)] automatically
        -- (it is needed for `legendreSym.mod` on line 227). Thus, we name the hypothesis
        -- `Nat.Prime p` explicitly on line 224 and prove `Fact (Nat.Prime p)` on line 225.
        /-
          a : Int
          b : Nat
          ⊢ ∀ (a_1 : Nat), Membership.mem b.primeFactorsList a_1 → ∀ (h₁ h₂ : Nat.Prime  …
        -/
        rintro p hp _ h₂
        /-
          a : Int
          b p : Nat
          hp : Membership.mem b.primeFactorsList p
          h₁✝ h₂ : Nat.Prime p
          ⊢ Eq (legendreSym p a) (legendreSym p (HMod.hMod a ↑b))
        -/
        letI : Fact p.Prime := ⟨h₂⟩
        conv_rhs =>
          rw [legendreSym.mod, Int.emod_emod_of_dvd _ (Int.natCast_dvd_natCast.2 <|
            dvd_of_mem_primeFactorsList hp), ← legendreSym.mod])


/-- The symbol `J(a | b)` depends only on `a` mod `b`. -/
theorem mod_left' {a₁ a₂ : ℤ} {b : ℕ} (h : a₁ % b = a₂ % b) : J(a₁ | b) = J(a₂ | b) := by
  /-
    a₁ a₂ : Int
    b : Nat
    h : Eq (HMod.hMod a₁ ↑b) (HMod.hMod a₂ ↑b)
    ⊢ Eq (jacobiSym a₁ b) (jacobiSym a₂ b)
  -/
  rw [mod_left, h, ← mod_left]
  /-
    🎉 no goals
  -/


/-- If `p` is prime, `J(a | p) = -1` and `p` divides `x^2 - a*y^2`, then `p` must divide
`x` and `y`. -/
theorem prime_dvd_of_eq_neg_one {p : ℕ} [Fact p.Prime] {a : ℤ} (h : J(a | p) = -1) {x y : ℤ}
    (hxy : ↑p ∣ (x ^ 2 - a * y ^ 2 : ℤ)) : ↑p ∣ x ∧ ↑p ∣ y := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : Int
    h : Eq (jacobiSym a p) (-1)
    x y : Int
    hxy : Dvd.dvd (↑p) (HSub.hSub (HPow.hPow x 2) (HMul.hMul a (HPow.hPow y 2)))
    ⊢ And (Dvd.dvd (↑p) x) (Dvd.dvd (↑p) y)
  -/
  rw [← legendreSym.to_jacobiSym] at h
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : Int
    h : Eq (legendreSym p a) (-1)
    x y : Int
    hxy : Dvd.dvd (↑p) (HSub.hSub (HPow.hPow x 2) (HMul.hMul a (HPow.hPow y 2)))
    ⊢ And (Dvd.dvd (↑p) x) (Dvd.dvd (↑p) y)
  -/
  exact legendreSym.prime_dvd_of_eq_neg_one h hxy
  /-
    🎉 no goals
  -/


/-- We can pull out a product over a list in the first argument of the Jacobi symbol. -/
theorem list_prod_left {l : List ℤ} {n : ℕ} : J(l.prod | n) = (l.map fun a => J(a | n)).prod := by
  /-
    l : List Int
    n : Nat
    ⊢ Eq (jacobiSym l.prod n) (List.map (fun a => jacobiSym a n) l).prod
  -/
  induction' l with n l' ih
    /-
      case nil
      n : Nat
      ⊢ Eq (jacobiSym List.nil.prod n) (List.map (fun a => jacobiSym a n) List.nil). …
    -/
  · simp only [List.prod_nil, List.map_nil, one_left]
    /-
      🎉 no goals
    -/
    /-
      case cons
      n✝ : Nat
      n : Int
      l' : List Int
      ih : Eq (jacobiSym l'.prod n✝) (List.map (fun a => jacobiSym a n✝) l').prod
      ⊢ Eq (jacobiSym (List.cons n l').prod n✝) (List.map (fun a => jacobiSym a n✝)  …
    -/
  · rw [List.map, List.prod_cons, List.prod_cons, mul_left, ih]
    /-
      🎉 no goals
    -/


/-- We can pull out a product over a list in the second argument of the Jacobi symbol. -/
theorem list_prod_right {a : ℤ} {l : List ℕ} (hl : ∀ n ∈ l, n ≠ 0) :
    J(a | l.prod) = (l.map fun n => J(a | n)).prod := by
  /-
    a : Int
    l : List Nat
    hl : ∀ (n : Nat), Membership.mem l n → Ne n 0
    ⊢ Eq (jacobiSym a l.prod) (List.map (fun n => jacobiSym a n) l).prod
  -/
  induction' l with n l' ih
    /-
      case nil
      a : Int
      hl : ∀ (n : Nat), Membership.mem List.nil n → Ne n 0
      ⊢ Eq (jacobiSym a List.nil.prod) (List.map (fun n => jacobiSym a n) List.nil). …
    -/
  · simp only [List.prod_nil, one_right, List.map_nil]
    /-
      🎉 no goals
    -/
    /-
      case cons
      a : Int
      n : Nat
      l' : List Nat
      ih : (∀ (n : Nat), Membership.mem l' n → Ne n 0) → Eq (jacobiSym a l'.prod) (L …
      hl : ∀ (n_1 : Nat), Membership.mem (List.cons n l') n_1 → Ne n_1 0
      ⊢ Eq (jacobiSym a (List.cons n l').prod) (List.map (fun n => jacobiSym a n) (L …
    -/
  · have hn := hl n (List.mem_cons_self n l')
    -- `n ≠ 0`
    /-
      case cons
      a : Int
      n : Nat
      l' : List Nat
      ih : (∀ (n : Nat), Membership.mem l' n → Ne n 0) → Eq (jacobiSym a l'.prod) (L …
      hl : ∀ (n_1 : Nat), Membership.mem (List.cons n l') n_1 → Ne n_1 0
      hn : Ne n 0
      ⊢ Eq (jacobiSym a (List.cons n l').prod) (List.map (fun n => jacobiSym a n) (L …
    -/
    have hl' := List.prod_ne_zero fun hf => hl 0 (List.mem_cons_of_mem _ hf) rfl
    -- `l'.prod ≠ 0`
    /-
      case cons
      a : Int
      n : Nat
      l' : List Nat
      ih : (∀ (n : Nat), Membership.mem l' n → Ne n 0) → Eq (jacobiSym a l'.prod) (L …
      hl : ∀ (n_1 : Nat), Membership.mem (List.cons n l') n_1 → Ne n_1 0
      hn : Ne n 0
      hl' : Ne l'.prod 0
      ⊢ Eq (jacobiSym a (List.cons n l').prod) (List.map (fun n => jacobiSym a n) (L …
    -/
    have h := fun m hm => hl m (List.mem_cons_of_mem _ hm)
    -- `∀ (m : ℕ), m ∈ l' → m ≠ 0`
    /-
      case cons
      a : Int
      n : Nat
      l' : List Nat
      ih : (∀ (n : Nat), Membership.mem l' n → Ne n 0) → Eq (jacobiSym a l'.prod) (L …
      hl : ∀ (n_1 : Nat), Membership.mem (List.cons n l') n_1 → Ne n_1 0
      hn : Ne n 0
      hl' : Ne l'.prod 0
      h : ∀ (m : Nat), Membership.mem l' m → Ne m 0
      ⊢ Eq (jacobiSym a (List.cons n l').prod) (List.map (fun n => jacobiSym a n) (L …
    -/
    rw [List.map, List.prod_cons, List.prod_cons, mul_right' a hn hl', ih h]
    /-
      🎉 no goals
    -/


/-- If `J(a | n) = -1`, then `n` has a prime divisor `p` such that `J(a | p) = -1`. -/
theorem eq_neg_one_at_prime_divisor_of_eq_neg_one {a : ℤ} {n : ℕ} (h : J(a | n) = -1) :
    ∃ p : ℕ, p.Prime ∧ p ∣ n ∧ J(a | p) = -1 := by
  have hn₀ : n ≠ 0 := by
    rintro rfl
    rw [zero_right, CharZero.eq_neg_self_iff] at h
    exact one_ne_zero h
  /-
    a : Int
    n : Nat
    h : Eq (jacobiSym a n) (-1)
    hn₀ : Ne n 0
    ⊢ Exists fun p => And (Nat.Prime p) (And (Dvd.dvd p n) (Eq (jacobiSym a p) (-1 …
  -/
  have hf₀ (p) (hp : p ∈ n.primeFactorsList) : p ≠ 0 := (Nat.pos_of_mem_primeFactorsList hp).ne.symm
  /-
    a : Int
    n : Nat
    h : Eq (jacobiSym a n) (-1)
    hn₀ : Ne n 0
    hf₀ : ∀ (p : Nat), Membership.mem n.primeFactorsList p → Ne p 0
    ⊢ Exists fun p => And (Nat.Prime p) (And (Dvd.dvd p n) (Eq (jacobiSym a p) (-1 …
  -/
  rw [← Nat.prod_primeFactorsList hn₀, list_prod_right hf₀] at h
  /-
    a : Int
    n : Nat
    h : Eq (List.map (fun n => jacobiSym a n) n.primeFactorsList).prod (-1)
    hn₀ : Ne n 0
    hf₀ : ∀ (p : Nat), Membership.mem n.primeFactorsList p → Ne p 0
    ⊢ Exists fun p => And (Nat.Prime p) (And (Dvd.dvd p n) (Eq (jacobiSym a p) (-1 …
  -/
  obtain ⟨p, hmem, hj⟩ := List.mem_map.mp (List.neg_one_mem_of_prod_eq_neg_one h)
  /-
    case intro.intro
    a : Int
    n : Nat
    h : Eq (List.map (fun n => jacobiSym a n) n.primeFactorsList).prod (-1)
    hn₀ : Ne n 0
    hf₀ : ∀ (p : Nat), Membership.mem n.primeFactorsList p → Ne p 0
    p : Nat
    hmem : Membership.mem n.primeFactorsList p
    hj : Eq (jacobiSym a p) (-1)
    ⊢ Exists fun p => And (Nat.Prime p) (And (Dvd.dvd p n) (Eq (jacobiSym a p) (-1 …
  -/
  exact ⟨p, Nat.prime_of_mem_primeFactorsList hmem, Nat.dvd_of_mem_primeFactorsList hmem, hj⟩
  /-
    🎉 no goals
  -/


/-- If `J(a | b)` is `-1`, then `a` is not a square modulo `b`. -/
theorem nonsquare_of_jacobiSym_eq_neg_one {a : ℤ} {b : ℕ} (h : J(a | b) = -1) :
    ¬IsSquare (a : ZMod b) := fun ⟨r, ha⟩ => by
  /-
    a : Int
    b : Nat
    h : Eq (jacobiSym a b) (-1)
    x✝ : IsSquare ↑a
    r : ZMod b
    ha : Eq (↑a) (HMul.hMul r r)
    ⊢ False
  -/
  rw [← r.coe_valMinAbs, ← Int.cast_mul, intCast_eq_intCast_iff', ← sq] at ha
  /-
    a : Int
    b : Nat
    h : Eq (jacobiSym a b) (-1)
    x✝ : IsSquare ↑a
    r : ZMod b
    ha : Eq (HMod.hMod a ↑b) (HMod.hMod (HPow.hPow r.valMinAbs 2) ↑b)
    ⊢ False
  -/
  apply (by norm_num : ¬(0 : ℤ) ≤ -1)
  /-
    a : Int
    b : Nat
    h : Eq (jacobiSym a b) (-1)
    x✝ : IsSquare ↑a
    r : ZMod b
    ha : Eq (HMod.hMod a ↑b) (HMod.hMod (HPow.hPow r.valMinAbs 2) ↑b)
    ⊢ LE.le 0 (-1)
  -/
  rw [← h, mod_left, ha, ← mod_left, pow_left]
  /-
    a : Int
    b : Nat
    h : Eq (jacobiSym a b) (-1)
    x✝ : IsSquare ↑a
    r : ZMod b
    ha : Eq (HMod.hMod a ↑b) (HMod.hMod (HPow.hPow r.valMinAbs 2) ↑b)
    ⊢ LE.le 0 (HPow.hPow (jacobiSym r.valMinAbs b) 2)
  -/
  apply sq_nonneg
  /-
    🎉 no goals
  -/


/-- If `p` is prime, then `J(a | p)` is `-1` iff `a` is not a square modulo `p`. -/
theorem nonsquare_iff_jacobiSym_eq_neg_one {a : ℤ} {p : ℕ} [Fact p.Prime] :
    J(a | p) = -1 ↔ ¬IsSquare (a : ZMod p) := by
  /-
    a : Int
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    ⊢ Iff (Eq (jacobiSym a p) (-1)) (Not (IsSquare ↑a))
  -/
  rw [← legendreSym.to_jacobiSym]
  /-
    a : Int
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    ⊢ Iff (Eq (legendreSym p a) (-1)) (Not (IsSquare ↑a))
  -/
  exact legendreSym.eq_neg_one_iff p
  /-
    🎉 no goals
  -/


/-- If `p` is prime and `J(a | p) = 1`, then `a` is a square mod `p`. -/
theorem isSquare_of_jacobiSym_eq_one {a : ℤ} {p : ℕ} [Fact p.Prime] (h : J(a | p) = 1) :
    IsSquare (a : ZMod p) :=
                             /-
                               a : Int
                               p : Nat
                               inst✝ : Fact (Nat.Prime p)
                               h : Eq (jacobiSym a p) 1
                               ⊢ Not (Not (IsSquare ↑a))
                             -/
  Classical.not_not.mp <| by rw [← nonsquare_iff_jacobiSym_eq_neg_one, h]; decide
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


/-- If `χ` is a multiplicative function such that `J(a | p) = χ p` for all odd primes `p`,
then `J(a | b)` equals `χ b` for all odd natural numbers `b`. -/
theorem value_at (a : ℤ) {R : Type*} [CommSemiring R] (χ : R →* ℤ)
    (hp : ∀ (p : ℕ) (pp : p.Prime), p ≠ 2 → @legendreSym p ⟨pp⟩ a = χ p) {b : ℕ} (hb : Odd b) :
    J(a | b) = χ b := by
  /-
    a : Int
    R : Type u_1
    inst✝ : CommSemiring R
    χ : MonoidHom R Int
    hp : ∀ (p : Nat) (pp : Nat.Prime p), Ne p 2 → Eq (legendreSym p a) (χ ↑p)
    b : Nat
    hb : Odd b
    ⊢ Eq (jacobiSym a b) (χ ↑b)
  -/
  conv_rhs => rw [← prod_primeFactorsList hb.pos.ne', cast_list_prod, map_list_prod χ]
  rw [jacobiSym, List.map_map, ← List.pmap_eq_map Nat.Prime _ _
    fun _ => prime_of_mem_primeFactorsList]
  /-
    a : Int
    R : Type u_1
    inst✝ : CommSemiring R
    χ : MonoidHom R Int
    hp : ∀ (p : Nat) (pp : Nat.Prime p), Ne p 2 → Eq (legendreSym p a) (χ ↑p)
    b : Nat
    hb : Odd b
    ⊢ Eq (List.pmap (fun p pp => legendreSym p a) b.primeFactorsList ⋯).prod (List …
  -/
  congr 1; apply List.pmap_congr_left
  /-
    case e_a.h
    a : Int
    R : Type u_1
    inst✝ : CommSemiring R
    χ : MonoidHom R Int
    hp : ∀ (p : Nat) (pp : Nat.Prime p), Ne p 2 → Eq (legendreSym p a) (χ ↑p)
    b : Nat
    hb : Odd b
    ⊢ ∀ (a_1 : Nat), Membership.mem b.primeFactorsList a_1 → ∀ (h₁ : Nat.Prime a_1 …
  -/
  exact fun p h pp _ => hp p pp (hb.ne_two_of_dvd_nat <| dvd_of_mem_primeFactorsList h)
  /-
    🎉 no goals
  -/


/-- If `b` is odd, then `J(-1 | b)` is given by `χ₄ b`. -/
theorem at_neg_one {b : ℕ} (hb : Odd b) : J(-1 | b) = χ₄ b :=
  -- Porting note: In mathlib3, it was written `χ₄` and Lean could guess that it had to use
  -- `χ₄.to_monoid_hom`. This is not the case with Lean 4.
  value_at (-1) χ₄.toMonoidHom (fun p pp => @legendreSym.at_neg_one p ⟨pp⟩) hb


/-- If `b` is odd, then `J(-a | b) = χ₄ b * J(a | b)`. -/
protected theorem neg (a : ℤ) {b : ℕ} (hb : Odd b) : J(-a | b) = χ₄ b * J(a | b) := by
  /-
    a : Int
    b : Nat
    hb : Odd b
    ⊢ Eq (jacobiSym (Neg.neg a) b) (HMul.hMul (ZMod.χ₄ ↑b) (jacobiSym a b))
  -/
  rw [neg_eq_neg_one_mul, mul_left, at_neg_one hb]
  /-
    🎉 no goals
  -/


/-- If `b` is odd, then `J(2 | b)` is given by `χ₈ b`. -/
theorem at_two {b : ℕ} (hb : Odd b) : J(2 | b) = χ₈ b :=
  value_at 2 χ₈.toMonoidHom (fun p pp => @legendreSym.at_two p ⟨pp⟩) hb


/-- If `b` is odd, then `J(-2 | b)` is given by `χ₈' b`. -/
theorem at_neg_two {b : ℕ} (hb : Odd b) : J(-2 | b) = χ₈' b :=
  value_at (-2) χ₈'.toMonoidHom (fun p pp => @legendreSym.at_neg_two p ⟨pp⟩) hb


theorem div_four_left {a : ℤ} {b : ℕ} (ha4 : a % 4 = 0) (hb2 : b % 2 = 1) :
    J(a / 4 | b) = J(a | b) := by
  /-
    a : Int
    b : Nat
    ha4 : Eq (HMod.hMod a 4) 0
    hb2 : Eq (HMod.hMod b 2) 1
    ⊢ Eq (jacobiSym (HDiv.hDiv a 4) b) (jacobiSym a b)
  -/
  obtain ⟨a, rfl⟩ := Int.dvd_of_emod_eq_zero ha4
  have : Int.gcd (2 : ℕ) b = 1 := by
    rw [Int.gcd_natCast_natCast, ← b.mod_add_div 2, hb2, Nat.gcd_add_mul_left_right,
      Nat.gcd_one_right]
  rw [Int.mul_ediv_cancel_left _ (by decide), jacobiSym.mul_left,
    (by decide : (4 : ℤ) = (2 : ℕ) ^ 2), jacobiSym.sq_one' this, one_mul]


theorem even_odd {a : ℤ} {b : ℕ} (ha2 : a % 2 = 0) (hb2 : b % 2 = 1) :
    (if b % 8 = 3 ∨ b % 8 = 5 then -J(a / 2 | b) else J(a / 2 | b)) = J(a | b) := by
  /-
    a : Int
    b : Nat
    ha2 : Eq (HMod.hMod a 2) 0
    hb2 : Eq (HMod.hMod b 2) 1
    ⊢ Eq (ite (Or (Eq (HMod.hMod b 8) 3) (Eq (HMod.hMod b 8) 5)) (Neg.neg (jacobiS …
  -/
  obtain ⟨a, rfl⟩ := Int.dvd_of_emod_eq_zero ha2
  rw [Int.mul_ediv_cancel_left _ (by decide), jacobiSym.mul_left,
    jacobiSym.at_two (Nat.odd_iff.mpr hb2), ZMod.χ₈_nat_eq_if_mod_eight,
    if_neg (Nat.mod_two_ne_zero.mpr hb2)]
  /-
    case intro
    b : Nat
    hb2 : Eq (HMod.hMod b 2) 1
    a : Int
    ha2 : Eq (HMod.hMod (HMul.hMul 2 a) 2) 0
    ⊢ Eq (ite (Or (Eq (HMod.hMod b 8) 3) (Eq (HMod.hMod b 8) 5)) (Neg.neg (jacobiS …
  -/
  have := Nat.mod_lt b (by decide : 0 < 8)
  /-
    case intro
    b : Nat
    hb2 : Eq (HMod.hMod b 2) 1
    a : Int
    ha2 : Eq (HMod.hMod (HMul.hMul 2 a) 2) 0
    this : LT.lt (HMod.hMod b 8) 8
    ⊢ Eq (ite (Or (Eq (HMod.hMod b 8) 3) (Eq (HMod.hMod b 8) 5)) (Neg.neg (jacobiS …
  -/
                               /-
                                 🎉 no goals
                               -/
                               /-
                                 🎉 no goals
                               -/
                               /-
                                 🎉 no goals
                               -/
  interval_cases h : b % 8 <;> simp_all <;>
                               /-
                                 🎉 no goals
                               -/
    /-
      case intro.«0»
      b : Nat
      hb2 : Eq (HMod.hMod b 2) 1
      a : Int
      h : Eq (HMod.hMod b 8) 0
      ⊢ Eq (jacobiSym a b) (Neg.neg (jacobiSym a b))
    -/
    /-
      case intro.«0»
      b : Nat
      hb2 : Eq (HMod.hMod b 2) 1
      a : Int
      h : Eq (HMod.hMod b 8) 0
      this : Eq (HMod.hMod 0 2) 1
      ⊢ Eq (jacobiSym a b) (Neg.neg (jacobiSym a b))
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      case intro.«6»
      b : Nat
      hb2 : Eq (HMod.hMod b 2) 1
      a : Int
      h : Eq (HMod.hMod b 8) 6
      this : Eq (HMod.hMod 6 2) 1
      ⊢ Eq (jacobiSym a b) (Neg.neg (jacobiSym a b))
    -/
    simp_all
    /-
      🎉 no goals
    -/


/-- The bi-multiplicative map giving the sign in the Law of Quadratic Reciprocity -/
def qrSign (m n : ℕ) : ℤ :=
  J(χ₄ m | n)


/-- We can express `qrSign m n` as a power of `-1` when `m` and `n` are odd. -/
theorem neg_one_pow {m n : ℕ} (hm : Odd m) (hn : Odd n) :
    qrSign m n = (-1) ^ (m / 2 * (n / 2)) := by
  /-
    m n : Nat
    hm : Odd m
    hn : Odd n
    ⊢ Eq (qrSign m n) (HPow.hPow (-1) (HMul.hMul (HDiv.hDiv m 2) (HDiv.hDiv n 2)))
  -/
  rw [qrSign, pow_mul, ← χ₄_eq_neg_one_pow (odd_iff.mp hm)]
  /-
    m n : Nat
    hm : Odd m
    hn : Odd n
    ⊢ Eq (jacobiSym (ZMod.χ₄ ↑m) n) (HPow.hPow (ZMod.χ₄ ↑m) (HDiv.hDiv n 2))
  -/
  cases' odd_mod_four_iff.mp (odd_iff.mp hm) with h h
    /-
      case inl
      m n : Nat
      hm : Odd m
      hn : Odd n
      h : Eq (HMod.hMod m 4) 1
      ⊢ Eq (jacobiSym (ZMod.χ₄ ↑m) n) (HPow.hPow (ZMod.χ₄ ↑m) (HDiv.hDiv n 2))
    -/
  · rw [χ₄_nat_one_mod_four h, jacobiSym.one_left, one_pow]
    /-
      🎉 no goals
    -/
    /-
      case inr
      m n : Nat
      hm : Odd m
      hn : Odd n
      h : Eq (HMod.hMod m 4) 3
      ⊢ Eq (jacobiSym (ZMod.χ₄ ↑m) n) (HPow.hPow (ZMod.χ₄ ↑m) (HDiv.hDiv n 2))
    -/
  · rw [χ₄_nat_three_mod_four h, ← χ₄_eq_neg_one_pow (odd_iff.mp hn), jacobiSym.at_neg_one hn]
    /-
      🎉 no goals
    -/


/-- When `m` and `n` are odd, then the square of `qrSign m n` is `1`. -/
theorem sq_eq_one {m n : ℕ} (hm : Odd m) (hn : Odd n) : qrSign m n ^ 2 = 1 := by
  /-
    m n : Nat
    hm : Odd m
    hn : Odd n
    ⊢ Eq (HPow.hPow (qrSign m n) 2) 1
  -/
  rw [neg_one_pow hm hn, ← pow_mul, mul_comm, pow_mul, neg_one_sq, one_pow]
  /-
    🎉 no goals
  -/


/-- `qrSign` is multiplicative in the first argument. -/
theorem mul_left (m₁ m₂ n : ℕ) : qrSign (m₁ * m₂) n = qrSign m₁ n * qrSign m₂ n := by
  /-
    m₁ m₂ n : Nat
    ⊢ Eq (qrSign (HMul.hMul m₁ m₂) n) (HMul.hMul (qrSign m₁ n) (qrSign m₂ n))
  -/
  simp_rw [qrSign, Nat.cast_mul, map_mul, jacobiSym.mul_left]
  /-
    🎉 no goals
  -/


/-- `qrSign` is multiplicative in the second argument. -/
theorem mul_right (m n₁ n₂ : ℕ) [NeZero n₁] [NeZero n₂] :
    qrSign m (n₁ * n₂) = qrSign m n₁ * qrSign m n₂ :=
  jacobiSym.mul_right (χ₄ m) n₁ n₂


/-- `qrSign` is symmetric when both arguments are odd. -/
protected theorem symm {m n : ℕ} (hm : Odd m) (hn : Odd n) : qrSign m n = qrSign n m := by
  /-
    m n : Nat
    hm : Odd m
    hn : Odd n
    ⊢ Eq (qrSign m n) (qrSign n m)
  -/
  rw [neg_one_pow hm hn, neg_one_pow hn hm, mul_comm (m / 2)]
  /-
    🎉 no goals
  -/


/-- We can move `qrSign m n` from one side of an equality to the other when `m` and `n` are odd. -/
theorem eq_iff_eq {m n : ℕ} (hm : Odd m) (hn : Odd n) (x y : ℤ) :
    qrSign m n * x = y ↔ x = qrSign m n * y := by
  refine
      ⟨fun h' =>
        let h := h'.symm
        ?_,
        fun h => ?_⟩ <;>
    /-
      case refine_1
      m n : Nat
      hm : Odd m
      hn : Odd n
      x y : Int
      h' : Eq (HMul.hMul (qrSign m n) x) y
      h : Eq y (HMul.hMul (qrSign m n) x) := Eq.symm h'
      ⊢ Eq x (HMul.hMul (qrSign m n) y)
    -/
    /-
      🎉 no goals
    -/
    rw [h, ← mul_assoc, ← pow_two, sq_eq_one hm hn, one_mul]
    /-
      🎉 no goals
    -/


/-- The **Law of Quadratic Reciprocity for the Jacobi symbol**, version with `qrSign` -/
theorem quadratic_reciprocity' {a b : ℕ} (ha : Odd a) (hb : Odd b) :
    J(a | b) = qrSign b a * J(b | a) := by
  -- define the right hand side for fixed `a` as a `ℕ →* ℤ`
  let rhs : ℕ → ℕ →* ℤ := fun a =>
    { toFun := fun x => qrSign x a * J(x | a)
      map_one' := by convert ← mul_one (M := ℤ) _; (on_goal 1 => symm); all_goals apply one_left
      map_mul' := fun x y => by
        -- Porting note: `simp_rw` on line 423 replaces `rw` to allow the rewrite rules to be
        -- applied under the binder `fun ↦ ...`
        simp_rw [qrSign.mul_left x y a, Nat.cast_mul, mul_left, mul_mul_mul_comm] }
  /-
    a b : Nat
    ha : Odd a
    hb : Odd b
    rhs : Nat → MonoidHom Nat Int := fun a => { toFun := fun x => HMul.hMul (qrSig …
    ⊢ Eq (jacobiSym (↑a) b) (HMul.hMul (qrSign b a) (jacobiSym (↑b) a))
  -/
  have rhs_apply : ∀ a b : ℕ, rhs a b = qrSign b a * J(b | a) := fun a b => rfl
  /-
    a b : Nat
    ha : Odd a
    hb : Odd b
    rhs : Nat → MonoidHom Nat Int := fun a => { toFun := fun x => HMul.hMul (qrSig …
    rhs_apply : ∀ (a b : Nat), Eq ((rhs a) b) (HMul.hMul (qrSign b a) (jacobiSym ( …
    ⊢ Eq (jacobiSym (↑a) b) (HMul.hMul (qrSign b a) (jacobiSym (↑b) a))
  -/
  refine value_at a (rhs a) (fun p pp hp => Eq.symm ?_) hb
  /-
    a b : Nat
    ha : Odd a
    hb : Odd b
    rhs : Nat → MonoidHom Nat Int := fun a => { toFun := fun x => HMul.hMul (qrSig …
    rhs_apply : ∀ (a b : Nat), Eq ((rhs a) b) (HMul.hMul (qrSign b a) (jacobiSym ( …
    p : Nat
    pp : Nat.Prime p
    hp : Ne p 2
    ⊢ Eq ((rhs a) ↑p) (legendreSym p ↑a)
  -/
  have hpo := pp.eq_two_or_odd'.resolve_left hp
  rw [@legendreSym.to_jacobiSym p ⟨pp⟩, rhs_apply, Nat.cast_id, qrSign.eq_iff_eq hpo ha,
    qrSign.symm hpo ha]
  /-
    a b : Nat
    ha : Odd a
    hb : Odd b
    rhs : Nat → MonoidHom Nat Int := fun a => { toFun := fun x => HMul.hMul (qrSig …
    rhs_apply : ∀ (a b : Nat), Eq ((rhs a) b) (HMul.hMul (qrSign b a) (jacobiSym ( …
    p : Nat
    pp : Nat.Prime p
    hp : Ne p 2
    hpo : Odd p
    ⊢ Eq (jacobiSym (↑p) a) (HMul.hMul (qrSign a p) (jacobiSym (↑a) p))
  -/
  refine value_at p (rhs p) (fun q pq hq => ?_) ha
  /-
    a b : Nat
    ha : Odd a
    hb : Odd b
    rhs : Nat → MonoidHom Nat Int := fun a => { toFun := fun x => HMul.hMul (qrSig …
    rhs_apply : ∀ (a b : Nat), Eq ((rhs a) b) (HMul.hMul (qrSign b a) (jacobiSym ( …
    p : Nat
    pp : Nat.Prime p
    hp : Ne p 2
    hpo : Odd p
    q : Nat
    pq : Nat.Prime q
    hq : Ne q 2
    ⊢ Eq (legendreSym q ↑p) ((rhs p) ↑q)
  -/
  have hqo := pq.eq_two_or_odd'.resolve_left hq
  rw [rhs_apply, Nat.cast_id, ← @legendreSym.to_jacobiSym p ⟨pp⟩, qrSign.symm hqo hpo,
    qrSign.neg_one_pow hpo hqo, @legendreSym.quadratic_reciprocity' p q ⟨pp⟩ ⟨pq⟩ hp hq]


/-- The Law of Quadratic Reciprocity for the Jacobi symbol -/
theorem quadratic_reciprocity {a b : ℕ} (ha : Odd a) (hb : Odd b) :
    J(a | b) = (-1) ^ (a / 2 * (b / 2)) * J(b | a) := by
  /-
    a b : Nat
    ha : Odd a
    hb : Odd b
    ⊢ Eq (jacobiSym (↑a) b) (HMul.hMul (HPow.hPow (-1) (HMul.hMul (HDiv.hDiv a 2)  …
  -/
  rw [← qrSign.neg_one_pow ha hb, qrSign.symm ha hb, quadratic_reciprocity' ha hb]
  /-
    🎉 no goals
  -/


/-- The Law of Quadratic Reciprocity for the Jacobi symbol: if `a` and `b` are natural numbers
with `a % 4 = 1` and `b` odd, then `J(a | b) = J(b | a)`. -/
theorem quadratic_reciprocity_one_mod_four {a b : ℕ} (ha : a % 4 = 1) (hb : Odd b) :
    J(a | b) = J(b | a) := by
  rw [quadratic_reciprocity (odd_iff.mpr (odd_of_mod_four_eq_one ha)) hb, pow_mul,
    neg_one_pow_div_two_of_one_mod_four ha, one_pow, one_mul]


/-- The Law of Quadratic Reciprocity for the Jacobi symbol: if `a` and `b` are natural numbers
with `a` odd and `b % 4 = 1`, then `J(a | b) = J(b | a)`. -/
theorem quadratic_reciprocity_one_mod_four' {a b : ℕ} (ha : Odd a) (hb : b % 4 = 1) :
    J(a | b) = J(b | a) :=
  (quadratic_reciprocity_one_mod_four hb ha).symm


/-- The Law of Quadratic Reciprocity for the Jacobi symbol: if `a` and `b` are natural numbers
both congruent to `3` mod `4`, then `J(a | b) = -J(b | a)`. -/
theorem quadratic_reciprocity_three_mod_four {a b : ℕ} (ha : a % 4 = 3) (hb : b % 4 = 3) :
    J(a | b) = -J(b | a) := by
  /-
    a b : Nat
    ha : Eq (HMod.hMod a 4) 3
    hb : Eq (HMod.hMod b 4) 3
    ⊢ Eq (jacobiSym (↑a) b) (Neg.neg (jacobiSym (↑b) a))
  -/
  let nop := @neg_one_pow_div_two_of_three_mod_four
  /-
    a b : Nat
    ha : Eq (HMod.hMod a 4) 3
    hb : Eq (HMod.hMod b 4) 3
    nop : ∀ {n : Nat}, Eq (HMod.hMod n 4) 3 → Eq (HPow.hPow (-1) (HDiv.hDiv n 2))  …
    ⊢ Eq (jacobiSym (↑a) b) (Neg.neg (jacobiSym (↑b) a))
  -/
  rw [quadratic_reciprocity, pow_mul, nop ha, nop hb, neg_one_mul] <;>
    /-
      case ha
      a b : Nat
      ha : Eq (HMod.hMod a 4) 3
      hb : Eq (HMod.hMod b 4) 3
      nop : ∀ {n : Nat}, Eq (HMod.hMod n 4) 3 → Eq (HPow.hPow (-1) (HDiv.hDiv n 2))  …
      ⊢ Odd a
    -/
    /-
      🎉 no goals
    -/
    rwa [odd_iff, odd_of_mod_four_eq_three]
    /-
      🎉 no goals
    -/


theorem quadratic_reciprocity_if {a b : ℕ} (ha2 : a % 2 = 1) (hb2 : b % 2 = 1) :
    (if a % 4 = 3 ∧ b % 4 = 3 then -J(b | a) else J(b | a)) = J(a | b) := by
  /-
    a b : Nat
    ha2 : Eq (HMod.hMod a 2) 1
    hb2 : Eq (HMod.hMod b 2) 1
    ⊢ Eq (ite (And (Eq (HMod.hMod a 4) 3) (Eq (HMod.hMod b 4) 3)) (Neg.neg (jacobi …
  -/
  rcases Nat.odd_mod_four_iff.mp ha2 with ha1 | ha3
    /-
      case inl
      a b : Nat
      ha2 : Eq (HMod.hMod a 2) 1
      hb2 : Eq (HMod.hMod b 2) 1
      ha1 : Eq (HMod.hMod a 4) 1
      ⊢ Eq (ite (And (Eq (HMod.hMod a 4) 3) (Eq (HMod.hMod b 4) 3)) (Neg.neg (jacobi …
    -/
  · simpa [ha1] using jacobiSym.quadratic_reciprocity_one_mod_four' (Nat.odd_iff.mpr hb2) ha1
    /-
      🎉 no goals
    -/
  /-
    case inr
    a b : Nat
    ha2 : Eq (HMod.hMod a 2) 1
    hb2 : Eq (HMod.hMod b 2) 1
    ha3 : Eq (HMod.hMod a 4) 3
    ⊢ Eq (ite (And (Eq (HMod.hMod a 4) 3) (Eq (HMod.hMod b 4) 3)) (Neg.neg (jacobi …
  -/
  rcases Nat.odd_mod_four_iff.mp hb2 with hb1 | hb3
    /-
      case inr.inl
      a b : Nat
      ha2 : Eq (HMod.hMod a 2) 1
      hb2 : Eq (HMod.hMod b 2) 1
      ha3 : Eq (HMod.hMod a 4) 3
      hb1 : Eq (HMod.hMod b 4) 1
      ⊢ Eq (ite (And (Eq (HMod.hMod a 4) 3) (Eq (HMod.hMod b 4) 3)) (Neg.neg (jacobi …
    -/
  · simpa [hb1] using jacobiSym.quadratic_reciprocity_one_mod_four hb1 (Nat.odd_iff.mpr ha2)
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    a b : Nat
    ha2 : Eq (HMod.hMod a 2) 1
    hb2 : Eq (HMod.hMod b 2) 1
    ha3 : Eq (HMod.hMod a 4) 3
    hb3 : Eq (HMod.hMod b 4) 3
    ⊢ Eq (ite (And (Eq (HMod.hMod a 4) 3) (Eq (HMod.hMod b 4) 3)) (Neg.neg (jacobi …
  -/
  simpa [ha3, hb3] using (jacobiSym.quadratic_reciprocity_three_mod_four ha3 hb3).symm
  /-
    🎉 no goals
  -/


/-- The Jacobi symbol `J(a | b)` depends only on `b` mod `4*a` (version for `a : ℕ`). -/
theorem mod_right' (a : ℕ) {b : ℕ} (hb : Odd b) : J(a | b) = J(a | b % (4 * a)) := by
  /-
    a b : Nat
    hb : Odd b
    ⊢ Eq (jacobiSym (↑a) b) (jacobiSym (↑a) (HMod.hMod b (HMul.hMul 4 a)))
  -/
  rcases eq_or_ne a 0 with (rfl | ha₀)
    /-
      case inl
      b : Nat
      hb : Odd b
      ⊢ Eq (jacobiSym (↑0) b) (jacobiSym (↑0) (HMod.hMod b (HMul.hMul 4 0)))
    -/
  · rw [mul_zero, mod_zero]
    /-
      🎉 no goals
    -/
  /-
    case inr
    a b : Nat
    hb : Odd b
    ha₀ : Ne a 0
    ⊢ Eq (jacobiSym (↑a) b) (jacobiSym (↑a) (HMod.hMod b (HMul.hMul 4 a)))
  -/
  have hb' : Odd (b % (4 * a)) := hb.mod_even (Even.mul_right (by decide) _)
  /-
    case inr
    a b : Nat
    hb : Odd b
    ha₀ : Ne a 0
    hb' : Odd (HMod.hMod b (HMul.hMul 4 a))
    ⊢ Eq (jacobiSym (↑a) b) (jacobiSym (↑a) (HMod.hMod b (HMul.hMul 4 a)))
  -/
  rcases exists_eq_pow_mul_and_not_dvd ha₀ 2 (by norm_num) with ⟨e, a', ha₁', ha₂⟩
  /-
    case inr.intro.intro.intro
    a b : Nat
    hb : Odd b
    ha₀ : Ne a 0
    hb' : Odd (HMod.hMod b (HMul.hMul 4 a))
    e a' : Nat
    ha₁' : Not (Dvd.dvd 2 a')
    ha₂ : Eq a (HMul.hMul (HPow.hPow 2 e) a')
    ⊢ Eq (jacobiSym (↑a) b) (jacobiSym (↑a) (HMod.hMod b (HMul.hMul 4 a)))
  -/
  have ha₁ := odd_iff.mpr (two_dvd_ne_zero.mp ha₁')
  /-
    case inr.intro.intro.intro
    a b : Nat
    hb : Odd b
    ha₀ : Ne a 0
    hb' : Odd (HMod.hMod b (HMul.hMul 4 a))
    e a' : Nat
    ha₁' : Not (Dvd.dvd 2 a')
    ha₂ : Eq a (HMul.hMul (HPow.hPow 2 e) a')
    ha₁ : Odd a'
    ⊢ Eq (jacobiSym (↑a) b) (jacobiSym (↑a) (HMod.hMod b (HMul.hMul 4 a)))
  -/
  nth_rw 2 [ha₂]; nth_rw 1 [ha₂]
  rw [Nat.cast_mul, mul_left, mul_left, quadratic_reciprocity' ha₁ hb,
    quadratic_reciprocity' ha₁ hb', Nat.cast_pow, pow_left, pow_left, Nat.cast_two, at_two hb,
    at_two hb']
  /-
    case inr.intro.intro.intro
    a b : Nat
    hb : Odd b
    ha₀ : Ne a 0
    hb' : Odd (HMod.hMod b (HMul.hMul 4 a))
    e a' : Nat
    ha₁' : Not (Dvd.dvd 2 a')
    ha₂ : Eq a (HMul.hMul (HPow.hPow 2 e) a')
    ha₁ : Odd a'
    ⊢ Eq (HMul.hMul (HPow.hPow (ZMod.χ₈ ↑b) e) (HMul.hMul (qrSign b a') (jacobiSym …
  -/
  congr 1; swap
    /-
      case inr.intro.intro.intro.e_a
      a b : Nat
      hb : Odd b
      ha₀ : Ne a 0
      hb' : Odd (HMod.hMod b (HMul.hMul 4 a))
      e a' : Nat
      ha₁' : Not (Dvd.dvd 2 a')
      ha₂ : Eq a (HMul.hMul (HPow.hPow 2 e) a')
      ha₁ : Odd a'
      ⊢ Eq (HMul.hMul (qrSign b a') (jacobiSym (↑b) a')) (HMul.hMul (qrSign (HMod.hM …
    -/
  · congr 1
      /-
        case inr.intro.intro.intro.e_a.e_a
        a b : Nat
        hb : Odd b
        ha₀ : Ne a 0
        hb' : Odd (HMod.hMod b (HMul.hMul 4 a))
        e a' : Nat
        ha₁' : Not (Dvd.dvd 2 a')
        ha₂ : Eq a (HMul.hMul (HPow.hPow 2 e) a')
        ha₁ : Odd a'
        ⊢ Eq (qrSign b a') (qrSign (HMod.hMod b (HMul.hMul 4 a)) a')
      -/
    · simp_rw [qrSign]
      /-
        case inr.intro.intro.intro.e_a.e_a
        a b : Nat
        hb : Odd b
        ha₀ : Ne a 0
        hb' : Odd (HMod.hMod b (HMul.hMul 4 a))
        e a' : Nat
        ha₁' : Not (Dvd.dvd 2 a')
        ha₂ : Eq a (HMul.hMul (HPow.hPow 2 e) a')
        ha₁ : Odd a'
        ⊢ Eq (jacobiSym (ZMod.χ₄ ↑b) a') (jacobiSym (ZMod.χ₄ ↑(HMod.hMod b (HMul.hMul  …
      -/
      rw [χ₄_nat_mod_four, χ₄_nat_mod_four (b % (4 * a)), mod_mod_of_dvd b (dvd_mul_right 4 a)]
      /-
        🎉 no goals
      -/
      /-
        case inr.intro.intro.intro.e_a.e_a
        a b : Nat
        hb : Odd b
        ha₀ : Ne a 0
        hb' : Odd (HMod.hMod b (HMul.hMul 4 a))
        e a' : Nat
        ha₁' : Not (Dvd.dvd 2 a')
        ha₂ : Eq a (HMul.hMul (HPow.hPow 2 e) a')
        ha₁ : Odd a'
        ⊢ Eq (jacobiSym (↑b) a') (jacobiSym (↑(HMod.hMod b (HMul.hMul 4 a))) a')
      -/
    · rw [mod_left ↑(b % _), mod_left b, Int.natCast_mod, Int.emod_emod_of_dvd b]
      /-
        case inr.intro.intro.intro.e_a.e_a
        a b : Nat
        hb : Odd b
        ha₀ : Ne a 0
        hb' : Odd (HMod.hMod b (HMul.hMul 4 a))
        e a' : Nat
        ha₁' : Not (Dvd.dvd 2 a')
        ha₂ : Eq a (HMul.hMul (HPow.hPow 2 e) a')
        ha₁ : Odd a'
        ⊢ Dvd.dvd ↑a' ↑(HMul.hMul 4 a)
      -/
      simp only [ha₂, Nat.cast_mul, ← mul_assoc]
      /-
        case inr.intro.intro.intro.e_a.e_a
        a b : Nat
        hb : Odd b
        ha₀ : Ne a 0
        hb' : Odd (HMod.hMod b (HMul.hMul 4 a))
        e a' : Nat
        ha₁' : Not (Dvd.dvd 2 a')
        ha₂ : Eq a (HMul.hMul (HPow.hPow 2 e) a')
        ha₁ : Odd a'
        ⊢ Dvd.dvd (↑a') (HMul.hMul (HMul.hMul ↑4 ↑(HPow.hPow 2 e)) ↑a')
      -/
      apply dvd_mul_left
      /-
        🎉 no goals
      -/
  -- Porting note: In mathlib3, it was written `cases' e`. In Lean 4, this resulted in the choice
  -- of a name other than e (for the case distinction of line 482) so we indicate the name
  -- to use explicitly.
  /-
    case inr.intro.intro.intro.e_a
    a b : Nat
    hb : Odd b
    ha₀ : Ne a 0
    hb' : Odd (HMod.hMod b (HMul.hMul 4 a))
    e a' : Nat
    ha₁' : Not (Dvd.dvd 2 a')
    ha₂ : Eq a (HMul.hMul (HPow.hPow 2 e) a')
    ha₁ : Odd a'
    ⊢ Eq (HPow.hPow (ZMod.χ₈ ↑b) e) (HPow.hPow (ZMod.χ₈ ↑(HMod.hMod b (HMul.hMul 4 …
  -/
  cases' e with e; · rfl
                     /-
                       🎉 no goals
                     -/
    /-
      case inr.intro.intro.intro.e_a.succ
      a b : Nat
      hb : Odd b
      ha₀ : Ne a 0
      hb' : Odd (HMod.hMod b (HMul.hMul 4 a))
      a' : Nat
      ha₁' : Not (Dvd.dvd 2 a')
      ha₁ : Odd a'
      e : Nat
      ha₂ : Eq a (HMul.hMul (HPow.hPow 2 (HAdd.hAdd e 1)) a')
      ⊢ Eq (HPow.hPow (ZMod.χ₈ ↑b) (HAdd.hAdd e 1)) (HPow.hPow (ZMod.χ₈ ↑(HMod.hMod  …
    -/
  · rw [χ₈_nat_mod_eight, χ₈_nat_mod_eight (b % (4 * a)), mod_mod_of_dvd b]
    /-
      case inr.intro.intro.intro.e_a.succ
      a b : Nat
      hb : Odd b
      ha₀ : Ne a 0
      hb' : Odd (HMod.hMod b (HMul.hMul 4 a))
      a' : Nat
      ha₁' : Not (Dvd.dvd 2 a')
      ha₁ : Odd a'
      e : Nat
      ha₂ : Eq a (HMul.hMul (HPow.hPow 2 (HAdd.hAdd e 1)) a')
      ⊢ Dvd.dvd 8 (HMul.hMul 4 a)
    -/
    use 2 ^ e * a'; rw [ha₂, Nat.pow_succ]; ring
                                            /-
                                              🎉 no goals
                                            -/


/-- The Jacobi symbol `J(a | b)` depends only on `b` mod `4*a`. -/
theorem mod_right (a : ℤ) {b : ℕ} (hb : Odd b) : J(a | b) = J(a | b % (4 * a.natAbs)) := by
  /-
    a : Int
    b : Nat
    hb : Odd b
    ⊢ Eq (jacobiSym a b) (jacobiSym a (HMod.hMod b (HMul.hMul 4 a.natAbs)))
  -/
  cases' Int.natAbs_eq a with ha ha <;> nth_rw 2 [ha] <;> nth_rw 1 [ha]
    /-
      case inl
      a : Int
      b : Nat
      hb : Odd b
      ha : Eq a ↑a.natAbs
      ⊢ Eq (jacobiSym (↑a.natAbs) b) (jacobiSym (↑a.natAbs) (HMod.hMod b (HMul.hMul  …
    -/
  · exact mod_right' a.natAbs hb
    /-
      🎉 no goals
    -/
    /-
      case inr
      a : Int
      b : Nat
      hb : Odd b
      ha : Eq a (Neg.neg ↑a.natAbs)
      ⊢ Eq (jacobiSym (Neg.neg ↑a.natAbs) b) (jacobiSym (Neg.neg ↑a.natAbs) (HMod.hM …
    -/
  · have hb' : Odd (b % (4 * a.natAbs)) := hb.mod_even (Even.mul_right (by decide) _)
    rw [jacobiSym.neg _ hb, jacobiSym.neg _ hb', mod_right' _ hb, χ₄_nat_mod_four,
      χ₄_nat_mod_four (b % (4 * _)), mod_mod_of_dvd b (dvd_mul_right 4 _)]


/-- Computes `J(a | b)` (or `-J(a | b)` if `flip` is set to `true`) given assumptions, by reducing
`a` to odd by repeated division and then using quadratic reciprocity to swap `a`, `b`. -/
private def fastJacobiSymAux (a b : ℕ) (flip : Bool) (ha0 : a > 0) : ℤ :=
  if ha4 : a % 4 = 0 then
    fastJacobiSymAux (a / 4) b flip
                                                                      /-
                                                                        a b : Nat
                                                                        flip : Bool
                                                                        ha0 : GT.gt a 0
                                                                        ha4 : Eq (HMod.hMod a 4) 0
                                                                        ⊢ LT.lt 0 4
                                                                      -/
      (a.div_pos (Nat.le_of_dvd ha0 (Nat.dvd_of_mod_eq_zero ha4)) (by decide))
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
  else if ha2 : a % 2 = 0 then
    fastJacobiSymAux (a / 2) b (xor (b % 8 = 3 ∨ b % 8 = 5) flip)
                                                                      /-
                                                                        a b : Nat
                                                                        flip : Bool
                                                                        ha0 : GT.gt a 0
                                                                        ha4 : Not (Eq (HMod.hMod a 4) 0)
                                                                        ha2 : Eq (HMod.hMod a 2) 0
                                                                        ⊢ LT.lt 0 2
                                                                      -/
      (a.div_pos (Nat.le_of_dvd ha0 (Nat.dvd_of_mod_eq_zero ha2)) (by decide))
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
  else if ha1 : a = 1 then
    if flip then -1 else 1
  else if hba : b % a = 0 then
    0
  else
    fastJacobiSymAux (b % a) a (xor (a % 4 = 3 ∧ b % 4 = 3) flip) (Nat.pos_of_ne_zero hba)
termination_by a
decreasing_by
  · exact a.div_lt_self ha0 (by decide)
  · exact a.div_lt_self ha0 (by decide)
  · exact b.mod_lt ha0


private theorem fastJacobiSymAux.eq_jacobiSym {a b : ℕ} {flip : Bool} {ha0 : a > 0}
    (hb2 : b % 2 = 1) (hb1 : b > 1) :
    fastJacobiSymAux a b flip ha0 = if flip then -J(a | b) else J(a | b) := by
  /-
    a b : Nat
    flip : Bool
    ha0 : GT.gt a 0
    hb2 : Eq (HMod.hMod b 2) 1
    hb1 : GT.gt b 1
    ⊢ Eq (fastJacobiSymAux a b flip ha0) (ite (Eq flip Bool.true) (Neg.neg (jacobi …
  -/
  induction' a using Nat.strongRecOn with a IH generalizing b flip
  /-
    case ind
    a : Nat
    IH : ∀ (m : Nat), LT.lt m a → ∀ {b : Nat} {flip : Bool} {ha0 : GT.gt m 0}, Eq  …
    b : Nat
    flip : Bool
    ha0 : GT.gt a 0
    hb2 : Eq (HMod.hMod b 2) 1
    hb1 : GT.gt b 1
    ⊢ Eq (fastJacobiSymAux a b flip ha0) (ite (Eq flip Bool.true) (Neg.neg (jacobi …
  -/
  unfold fastJacobiSymAux
  /-
    case ind
    a : Nat
    IH : ∀ (m : Nat), LT.lt m a → ∀ {b : Nat} {flip : Bool} {ha0 : GT.gt m 0}, Eq  …
    b : Nat
    flip : Bool
    ha0 : GT.gt a 0
    hb2 : Eq (HMod.hMod b 2) 1
    hb1 : GT.gt b 1
    ⊢ Eq (dite (Eq (HMod.hMod a 4) 0) (fun ha4 => fastJacobiSymAux (HDiv.hDiv a 4) …
  -/
  split <;> rename_i ha4
    /-
      case ind.isTrue
      a : Nat
      IH : ∀ (m : Nat), LT.lt m a → ∀ {b : Nat} {flip : Bool} {ha0 : GT.gt m 0}, Eq  …
      b : Nat
      flip : Bool
      ha0 : GT.gt a 0
      hb2 : Eq (HMod.hMod b 2) 1
      hb1 : GT.gt b 1
      ha4 : Eq (HMod.hMod a 4) 0
      ⊢ Eq (fastJacobiSymAux (HDiv.hDiv a 4) b flip ⋯) (ite (Eq flip Bool.true) (Neg …
    -/
  · rw [IH (a / 4) (a.div_lt_self ha0 (by decide)) hb2 hb1]
    /-
      case ind.isTrue
      a : Nat
      IH : ∀ (m : Nat), LT.lt m a → ∀ {b : Nat} {flip : Bool} {ha0 : GT.gt m 0}, Eq  …
      b : Nat
      flip : Bool
      ha0 : GT.gt a 0
      hb2 : Eq (HMod.hMod b 2) 1
      hb1 : GT.gt b 1
      ha4 : Eq (HMod.hMod a 4) 0
      ⊢ Eq (ite (Eq flip Bool.true) (Neg.neg (jacobiSym (↑(HDiv.hDiv a 4)) b)) (jaco …
    -/
    simp only [Int.ofNat_ediv, Nat.cast_ofNat, div_four_left (a := a) (mod_cast ha4) hb2]
    /-
      🎉 no goals
    -/
  /-
    case ind.isFalse
    a : Nat
    IH : ∀ (m : Nat), LT.lt m a → ∀ {b : Nat} {flip : Bool} {ha0 : GT.gt m 0}, Eq  …
    b : Nat
    flip : Bool
    ha0 : GT.gt a 0
    hb2 : Eq (HMod.hMod b 2) 1
    hb1 : GT.gt b 1
    ha4 : Not (Eq (HMod.hMod a 4) 0)
    ⊢ Eq (dite (Eq (HMod.hMod a 2) 0) (fun ha2 => fastJacobiSymAux (HDiv.hDiv a 2) …
  -/
  split <;> rename_i ha2
    /-
      case ind.isFalse.isTrue
      a : Nat
      IH : ∀ (m : Nat), LT.lt m a → ∀ {b : Nat} {flip : Bool} {ha0 : GT.gt m 0}, Eq  …
      b : Nat
      flip : Bool
      ha0 : GT.gt a 0
      hb2 : Eq (HMod.hMod b 2) 1
      hb1 : GT.gt b 1
      ha4 : Not (Eq (HMod.hMod a 4) 0)
      ha2 : Eq (HMod.hMod a 2) 0
      ⊢ Eq (fastJacobiSymAux (HDiv.hDiv a 2) b ((Decidable.decide (Or (Eq (HMod.hMod …
    -/
  · rw [IH (a / 2) (a.div_lt_self ha0 (by decide)) hb2 hb1]
    /-
      case ind.isFalse.isTrue
      a : Nat
      IH : ∀ (m : Nat), LT.lt m a → ∀ {b : Nat} {flip : Bool} {ha0 : GT.gt m 0}, Eq  …
      b : Nat
      flip : Bool
      ha0 : GT.gt a 0
      hb2 : Eq (HMod.hMod b 2) 1
      hb1 : GT.gt b 1
      ha4 : Not (Eq (HMod.hMod a 4) 0)
      ha2 : Eq (HMod.hMod a 2) 0
      ⊢ Eq (ite (Eq ((Decidable.decide (Or (Eq (HMod.hMod b 8) 3) (Eq (HMod.hMod b 8 …
    -/
    simp only [Int.ofNat_ediv, Nat.cast_ofNat, ← even_odd (a := a) (mod_cast ha2) hb2]
    /-
      case ind.isFalse.isTrue
      a : Nat
      IH : ∀ (m : Nat), LT.lt m a → ∀ {b : Nat} {flip : Bool} {ha0 : GT.gt m 0}, Eq  …
      b : Nat
      flip : Bool
      ha0 : GT.gt a 0
      hb2 : Eq (HMod.hMod b 2) 1
      hb1 : GT.gt b 1
      ha4 : Not (Eq (HMod.hMod a 4) 0)
      ha2 : Eq (HMod.hMod a 2) 0
      ⊢ Eq (ite (Eq ((Decidable.decide (Or (Eq (HMod.hMod b 8) 3) (Eq (HMod.hMod b 8 …
    -/
                                           /-
                                             🎉 no goals
                                           -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
    by_cases h : b % 8 = 3 ∨ b % 8 = 5 <;> simp [h]; cases flip <;> simp
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  /-
    case ind.isFalse.isFalse
    a : Nat
    IH : ∀ (m : Nat), LT.lt m a → ∀ {b : Nat} {flip : Bool} {ha0 : GT.gt m 0}, Eq  …
    b : Nat
    flip : Bool
    ha0 : GT.gt a 0
    hb2 : Eq (HMod.hMod b 2) 1
    hb1 : GT.gt b 1
    ha4 : Not (Eq (HMod.hMod a 4) 0)
    ha2 : Not (Eq (HMod.hMod a 2) 0)
    ⊢ Eq (dite (Eq a 1) (fun ha1 => ite (Eq flip Bool.true) (-1) 1) fun ha1 => dit …
  -/
  split <;> rename_i ha1
    /-
      case ind.isFalse.isFalse.isTrue
      a : Nat
      IH : ∀ (m : Nat), LT.lt m a → ∀ {b : Nat} {flip : Bool} {ha0 : GT.gt m 0}, Eq  …
      b : Nat
      flip : Bool
      ha0 : GT.gt a 0
      hb2 : Eq (HMod.hMod b 2) 1
      hb1 : GT.gt b 1
      ha4 : Not (Eq (HMod.hMod a 4) 0)
      ha2 : Not (Eq (HMod.hMod a 2) 0)
      ha1 : Eq a 1
      ⊢ Eq (ite (Eq flip Bool.true) (-1) 1) (ite (Eq flip Bool.true) (Neg.neg (jacob …
    -/
  · subst ha1; simp
               /-
                 🎉 no goals
               -/
  /-
    case ind.isFalse.isFalse.isFalse
    a : Nat
    IH : ∀ (m : Nat), LT.lt m a → ∀ {b : Nat} {flip : Bool} {ha0 : GT.gt m 0}, Eq  …
    b : Nat
    flip : Bool
    ha0 : GT.gt a 0
    hb2 : Eq (HMod.hMod b 2) 1
    hb1 : GT.gt b 1
    ha4 : Not (Eq (HMod.hMod a 4) 0)
    ha2 : Not (Eq (HMod.hMod a 2) 0)
    ha1 : Not (Eq a 1)
    ⊢ Eq (dite (Eq (HMod.hMod b a) 0) (fun hba => 0) fun hba => fastJacobiSymAux ( …
  -/
  split <;> rename_i hba
    /-
      case ind.isFalse.isFalse.isFalse.isTrue
      a : Nat
      IH : ∀ (m : Nat), LT.lt m a → ∀ {b : Nat} {flip : Bool} {ha0 : GT.gt m 0}, Eq  …
      b : Nat
      flip : Bool
      ha0 : GT.gt a 0
      hb2 : Eq (HMod.hMod b 2) 1
      hb1 : GT.gt b 1
      ha4 : Not (Eq (HMod.hMod a 4) 0)
      ha2 : Not (Eq (HMod.hMod a 2) 0)
      ha1 : Not (Eq a 1)
      hba : Eq (HMod.hMod b a) 0
      ⊢ Eq 0 (ite (Eq flip Bool.true) (Neg.neg (jacobiSym (↑a) b)) (jacobiSym (↑a) b))
    -/
  · suffices J(a | b) = 0 by simp [this]
    /-
      case ind.isFalse.isFalse.isFalse.isTrue
      a : Nat
      IH : ∀ (m : Nat), LT.lt m a → ∀ {b : Nat} {flip : Bool} {ha0 : GT.gt m 0}, Eq  …
      b : Nat
      flip : Bool
      ha0 : GT.gt a 0
      hb2 : Eq (HMod.hMod b 2) 1
      hb1 : GT.gt b 1
      ha4 : Not (Eq (HMod.hMod a 4) 0)
      ha2 : Not (Eq (HMod.hMod a 2) 0)
      ha1 : Not (Eq a 1)
      hba : Eq (HMod.hMod b a) 0
      ⊢ Eq (jacobiSym (↑a) b) 0
    -/
    refine eq_zero_iff.mpr ⟨fun h ↦ absurd (h ▸ hb1) (by decide), ?_⟩
    /-
      case ind.isFalse.isFalse.isFalse.isTrue
      a : Nat
      IH : ∀ (m : Nat), LT.lt m a → ∀ {b : Nat} {flip : Bool} {ha0 : GT.gt m 0}, Eq  …
      b : Nat
      flip : Bool
      ha0 : GT.gt a 0
      hb2 : Eq (HMod.hMod b 2) 1
      hb1 : GT.gt b 1
      ha4 : Not (Eq (HMod.hMod a 4) 0)
      ha2 : Not (Eq (HMod.hMod a 2) 0)
      ha1 : Not (Eq a 1)
      hba : Eq (HMod.hMod b a) 0
      ⊢ Ne ((↑a).gcd ↑b) 1
    -/
    rwa [Int.gcd_natCast_natCast, Nat.gcd_eq_left (Nat.dvd_of_mod_eq_zero hba)]
    /-
      🎉 no goals
    -/
  /-
    case ind.isFalse.isFalse.isFalse.isFalse
    a : Nat
    IH : ∀ (m : Nat), LT.lt m a → ∀ {b : Nat} {flip : Bool} {ha0 : GT.gt m 0}, Eq  …
    b : Nat
    flip : Bool
    ha0 : GT.gt a 0
    hb2 : Eq (HMod.hMod b 2) 1
    hb1 : GT.gt b 1
    ha4 : Not (Eq (HMod.hMod a 4) 0)
    ha2 : Not (Eq (HMod.hMod a 2) 0)
    ha1 : Not (Eq a 1)
    hba : Not (Eq (HMod.hMod b a) 0)
    ⊢ Eq (fastJacobiSymAux (HMod.hMod b a) a ((Decidable.decide (And (Eq (HMod.hMo …
  -/
  rw [IH (b % a) (b.mod_lt ha0) (Nat.mod_two_ne_zero.mp ha2) (lt_of_le_of_ne ha0 (Ne.symm ha1))]
  /-
    case ind.isFalse.isFalse.isFalse.isFalse
    a : Nat
    IH : ∀ (m : Nat), LT.lt m a → ∀ {b : Nat} {flip : Bool} {ha0 : GT.gt m 0}, Eq  …
    b : Nat
    flip : Bool
    ha0 : GT.gt a 0
    hb2 : Eq (HMod.hMod b 2) 1
    hb1 : GT.gt b 1
    ha4 : Not (Eq (HMod.hMod a 4) 0)
    ha2 : Not (Eq (HMod.hMod a 2) 0)
    ha1 : Not (Eq a 1)
    hba : Not (Eq (HMod.hMod b a) 0)
    ⊢ Eq (ite (Eq ((Decidable.decide (And (Eq (HMod.hMod a 4) 3) (Eq (HMod.hMod b  …
  -/
  simp only [Int.natCast_mod, ← mod_left]
  /-
    case ind.isFalse.isFalse.isFalse.isFalse
    a : Nat
    IH : ∀ (m : Nat), LT.lt m a → ∀ {b : Nat} {flip : Bool} {ha0 : GT.gt m 0}, Eq  …
    b : Nat
    flip : Bool
    ha0 : GT.gt a 0
    hb2 : Eq (HMod.hMod b 2) 1
    hb1 : GT.gt b 1
    ha4 : Not (Eq (HMod.hMod a 4) 0)
    ha2 : Not (Eq (HMod.hMod a 2) 0)
    ha1 : Not (Eq a 1)
    hba : Not (Eq (HMod.hMod b a) 0)
    ⊢ Eq (ite (Eq ((Decidable.decide (And (Eq (HMod.hMod a 4) 3) (Eq (HMod.hMod b  …
  -/
  rw [← quadratic_reciprocity_if (Nat.mod_two_ne_zero.mp ha2) hb2]
  /-
    case ind.isFalse.isFalse.isFalse.isFalse
    a : Nat
    IH : ∀ (m : Nat), LT.lt m a → ∀ {b : Nat} {flip : Bool} {ha0 : GT.gt m 0}, Eq  …
    b : Nat
    flip : Bool
    ha0 : GT.gt a 0
    hb2 : Eq (HMod.hMod b 2) 1
    hb1 : GT.gt b 1
    ha4 : Not (Eq (HMod.hMod a 4) 0)
    ha2 : Not (Eq (HMod.hMod a 2) 0)
    ha1 : Not (Eq a 1)
    hba : Not (Eq (HMod.hMod b a) 0)
    ⊢ Eq (ite (Eq ((Decidable.decide (And (Eq (HMod.hMod a 4) 3) (Eq (HMod.hMod b  …
  -/
                                         /-
                                           🎉 no goals
                                         -/
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
  by_cases h : a % 4 = 3 ∧ b % 4 = 3 <;> simp [h]; cases flip <;> simp
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- Computes `J(a | b)` by reducing `b` to odd by repeated division and then using
`fastJacobiSymAux`. -/
private def fastJacobiSym (a : ℤ) (b : ℕ) : ℤ :=
  if hb0 : b = 0 then
    1
  else if _ : b % 2 = 0 then
    if a % 2 = 0 then
      0
    else
      have : b / 2 < b := b.div_lt_self (Nat.pos_of_ne_zero hb0) one_lt_two
      fastJacobiSym a (b / 2)
  else if b = 1 then
    1
  else if hab : a % b = 0 then
    0
  else
    fastJacobiSymAux (a % b).natAbs b false (Int.natAbs_pos.mpr hab)


@[csimp] private theorem fastJacobiSym.eq : jacobiSym = fastJacobiSym := by
  /-
    ⊢ Eq jacobiSym fastJacobiSym
  -/
  ext a b
  /-
    case h.h
    a : Int
    b : Nat
    ⊢ Eq (jacobiSym a b) (fastJacobiSym a b)
  -/
  induction' b using Nat.strongRecOn with b IH
  /-
    case h.h.ind
    a : Int
    b : Nat
    IH : ∀ (m : Nat), LT.lt m b → Eq (jacobiSym a m) (fastJacobiSym a m)
    ⊢ Eq (jacobiSym a b) (fastJacobiSym a b)
  -/
  unfold fastJacobiSym
  /-
    case h.h.ind
    a : Int
    b : Nat
    IH : ∀ (m : Nat), LT.lt m b → Eq (jacobiSym a m) (fastJacobiSym a m)
    ⊢ Eq (jacobiSym a b) (dite (Eq b 0) (fun hb0 => 1) fun hb0 => dite (Eq (HMod.h …
  -/
  split_ifs with hb0 hb2 ha2 hb1 hab
    /-
      case pos
      a : Int
      b : Nat
      IH : ∀ (m : Nat), LT.lt m b → Eq (jacobiSym a m) (fastJacobiSym a m)
      hb0 : Eq b 0
      ⊢ Eq (jacobiSym a b) 1
    -/
  · rw [hb0, zero_right]
    /-
      🎉 no goals
    -/
    /-
      case pos
      a : Int
      b : Nat
      IH : ∀ (m : Nat), LT.lt m b → Eq (jacobiSym a m) (fastJacobiSym a m)
      hb0 : Not (Eq b 0)
      hb2 : Eq (HMod.hMod b 2) 0
      ha2 : Eq (HMod.hMod a 2) 0
      ⊢ Eq (jacobiSym a b) 0
    -/
  · refine eq_zero_iff.mpr ⟨hb0, ne_of_gt ?_⟩
    /-
      case pos
      a : Int
      b : Nat
      IH : ∀ (m : Nat), LT.lt m b → Eq (jacobiSym a m) (fastJacobiSym a m)
      hb0 : Not (Eq b 0)
      hb2 : Eq (HMod.hMod b 2) 0
      ha2 : Eq (HMod.hMod a 2) 0
      ⊢ LT.lt 1 (a.gcd ↑b)
    -/
    refine Nat.le_of_dvd (Int.gcd_pos_iff.mpr (mod_cast .inr hb0)) ?_
    /-
      case pos
      a : Int
      b : Nat
      IH : ∀ (m : Nat), LT.lt m b → Eq (jacobiSym a m) (fastJacobiSym a m)
      hb0 : Not (Eq b 0)
      hb2 : Eq (HMod.hMod b 2) 0
      ha2 : Eq (HMod.hMod a 2) 0
      ⊢ Dvd.dvd (Nat.succ 1) (a.gcd ↑b)
    -/
    refine Nat.dvd_gcd (Int.ofNat_dvd_left.mp (Int.dvd_of_emod_eq_zero ha2)) ?_
    /-
      case pos
      a : Int
      b : Nat
      IH : ∀ (m : Nat), LT.lt m b → Eq (jacobiSym a m) (fastJacobiSym a m)
      hb0 : Not (Eq b 0)
      hb2 : Eq (HMod.hMod b 2) 0
      ha2 : Eq (HMod.hMod a 2) 0
      ⊢ Dvd.dvd (Nat.succ 1) (↑b).natAbs
    -/
    exact Int.ofNat_dvd_left.mp (Int.dvd_of_emod_eq_zero (mod_cast hb2))
    /-
      🎉 no goals
    -/
    /-
      case neg
      a : Int
      b : Nat
      IH : ∀ (m : Nat), LT.lt m b → Eq (jacobiSym a m) (fastJacobiSym a m)
      hb0 : Not (Eq b 0)
      hb2 : Eq (HMod.hMod b 2) 0
      ha2 : Not (Eq (HMod.hMod a 2) 0)
      ⊢ Eq (jacobiSym a b) (fastJacobiSym a (HDiv.hDiv b 2))
    -/
  · rw [← IH (b / 2) (b.div_lt_self (Nat.pos_of_ne_zero hb0) one_lt_two)]
    /-
      case neg
      a : Int
      b : Nat
      IH : ∀ (m : Nat), LT.lt m b → Eq (jacobiSym a m) (fastJacobiSym a m)
      hb0 : Not (Eq b 0)
      hb2 : Eq (HMod.hMod b 2) 0
      ha2 : Not (Eq (HMod.hMod a 2) 0)
      ⊢ Eq (jacobiSym a b) (jacobiSym a (HDiv.hDiv b 2))
    -/
    obtain ⟨b, rfl⟩ := Nat.dvd_of_mod_eq_zero hb2
    rw [mul_right' a (by decide) fun h ↦ hb0 (mul_eq_zero_of_right 2 h),
      b.mul_div_cancel_left (by decide), mod_left a 2, Nat.cast_ofNat,
      Int.emod_two_ne_zero.mp ha2, one_left, one_mul]
    /-
      case pos
      a : Int
      b : Nat
      IH : ∀ (m : Nat), LT.lt m b → Eq (jacobiSym a m) (fastJacobiSym a m)
      hb0 : Not (Eq b 0)
      hb2 : Not (Eq (HMod.hMod b 2) 0)
      hb1 : Eq b 1
      ⊢ Eq (jacobiSym a b) 1
    -/
  · rw [hb1, one_right]
    /-
      🎉 no goals
    -/
    /-
      case pos
      a : Int
      b : Nat
      IH : ∀ (m : Nat), LT.lt m b → Eq (jacobiSym a m) (fastJacobiSym a m)
      hb0 : Not (Eq b 0)
      hb2 : Not (Eq (HMod.hMod b 2) 0)
      hb1 : Not (Eq b 1)
      hab : Eq (HMod.hMod a ↑b) 0
      ⊢ Eq (jacobiSym a b) 0
    -/
  · rw [mod_left, hab, zero_left (lt_of_le_of_ne (Nat.pos_of_ne_zero hb0) (Ne.symm hb1))]
    /-
      🎉 no goals
    -/
  · rw [fastJacobiSymAux.eq_jacobiSym, if_neg Bool.false_ne_true, mod_left a b,
      Int.natAbs_of_nonneg (a.emod_nonneg (mod_cast hb0))]
      /-
        case neg.hb2
        a : Int
        b : Nat
        IH : ∀ (m : Nat), LT.lt m b → Eq (jacobiSym a m) (fastJacobiSym a m)
        hb0 : Not (Eq b 0)
        hb2 : Not (Eq (HMod.hMod b 2) 0)
        hb1 : Not (Eq b 1)
        hab : Not (Eq (HMod.hMod a ↑b) 0)
        ⊢ Eq (HMod.hMod b 2) 1
      -/
    · exact Nat.mod_two_ne_zero.mp hb2
      /-
        🎉 no goals
      -/
      /-
        case neg.hb1
        a : Int
        b : Nat
        IH : ∀ (m : Nat), LT.lt m b → Eq (jacobiSym a m) (fastJacobiSym a m)
        hb0 : Not (Eq b 0)
        hb2 : Not (Eq (HMod.hMod b 2) 0)
        hb1 : Not (Eq b 1)
        hab : Not (Eq (HMod.hMod a ↑b) 0)
        ⊢ GT.gt b 1
      -/
    · exact lt_of_le_of_ne (Nat.one_le_iff_ne_zero.mpr hb0) (Ne.symm hb1)
      /-
        🎉 no goals
      -/


/-- Computes `legendreSym p a` using `fastJacobiSym`. -/
@[inline, nolint unusedArguments]
private def fastLegendreSym (p : ℕ) [Fact p.Prime] (a : ℤ) : ℤ := J(a | p)


@[csimp] private theorem fastLegendreSym.eq : legendreSym = fastLegendreSym := by
  /-
    ⊢ Eq legendreSym fastLegendreSym
  -/
  ext p _ a; rw [legendreSym.to_jacobiSym, fastLegendreSym]
             /-
               🎉 no goals
             -/


