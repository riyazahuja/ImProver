                                                                          /-
                                                                            α : Type u_3
                                                                            inst✝ : DivisionSemiring α
                                                                            n : Nat
                                                                            ⊢ Eq ↑↑n ↑n
                                                                          -/
@[simp, norm_cast] lemma cast_natCast (n : ℕ) : ((n : ℚ≥0) : α) = n := by simp [cast_def]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[simp, norm_cast] lemma cast_ofNat (n : ℕ) [n.AtLeastTwo] :
    (ofNat(n) : ℚ≥0) = (ofNat(n) : α) := cast_natCast _


@[simp, norm_cast] lemma cast_zero : ((0 : ℚ≥0) : α) = 0 := (cast_natCast _).trans Nat.cast_zero

@[simp, norm_cast] lemma cast_one : ((1 : ℚ≥0) : α) = 1 := (cast_natCast _).trans Nat.cast_one


lemma cast_commute (q : ℚ≥0) (a : α) : Commute (↑q) a := by
  /-
    α : Type u_3
    inst✝ : DivisionSemiring α
    q : NNRat
    a : α
    ⊢ Commute (↑q) a
  -/
  simpa only [cast_def] using (q.num.cast_commute a).div_left (q.den.cast_commute a)
  /-
    🎉 no goals
  -/


lemma commute_cast (a : α) (q : ℚ≥0) : Commute a q := (cast_commute ..).symm


lemma cast_comm (q : ℚ≥0) (a : α) : q * a = a * q := cast_commute _ _


@[norm_cast] lemma cast_divNat_of_ne_zero (a : ℕ) {b : ℕ} (hb : (b : α) ≠ 0) :
    divNat a b = (a / b : α) := by
  /-
    α : Type u_3
    inst✝ : DivisionSemiring α
    a b : Nat
    hb : Ne (↑b) 0
    ⊢ Eq (↑(NNRat.divNat a b)) (HDiv.hDiv ↑a ↑b)
  -/
  rcases e : divNat a b with ⟨⟨n, d, h, c⟩, hn⟩
  /-
    case mk.mk'
    α : Type u_3
    inst✝ : DivisionSemiring α
    a b : Nat
    hb : Ne (↑b) 0
    n : Int
    d : Nat
    h : Ne d 0
    c : n.natAbs.Coprime d
    hn : LE.le 0 { num := n, den := d, den_nz := h, reduced := c }
    e : Eq (NNRat.divNat a b) ⟨{ num := n, den := d, den_nz := h, reduced := c },  …
    ⊢ Eq (↑⟨{ num := n, den := d, den_nz := h, reduced := c }, hn⟩) (HDiv.hDiv ↑a  …
  -/
  rw [← Rat.num_nonneg] at hn
  /-
    case mk.mk'
    α : Type u_3
    inst✝ : DivisionSemiring α
    a b : Nat
    hb : Ne (↑b) 0
    n : Int
    d : Nat
    h : Ne d 0
    c : n.natAbs.Coprime d
    hn✝ : LE.le 0 { num := n, den := d, den_nz := h, reduced := c }
    hn : LE.le 0 { num := n, den := d, den_nz := h, reduced := c }.num
    e : Eq (NNRat.divNat a b) ⟨{ num := n, den := d, den_nz := h, reduced := c },  …
    ⊢ Eq (↑⟨{ num := n, den := d, den_nz := h, reduced := c }, hn✝⟩) (HDiv.hDiv ↑a …
  -/
  lift n to ℕ using hn
  have hd : (d : α) ≠ 0 := by
    refine fun hd ↦ hb ?_
    have : Rat.divInt a b = _ := congr_arg NNRat.cast e
    obtain ⟨k, rfl⟩ : d ∣ b := by simpa [Int.natCast_dvd_natCast, this] using Rat.den_dvd a b
    simp [*]
  /-
    case mk.mk'.intro
    α : Type u_3
    inst✝ : DivisionSemiring α
    a b : Nat
    hb : Ne (↑b) 0
    d : Nat
    h : Ne d 0
    n : Nat
    c : (↑n).natAbs.Coprime d
    hn : LE.le 0 { num := ↑n, den := d, den_nz := h, reduced := c }
    e : Eq (NNRat.divNat a b) ⟨{ num := ↑n, den := d, den_nz := h, reduced := c }, …
    hd : Ne (↑d) 0
    ⊢ Eq (↑⟨{ num := ↑n, den := d, den_nz := h, reduced := c }, hn⟩) (HDiv.hDiv ↑a …
  -/
  have hb' : b ≠ 0 := by rintro rfl; exact hb Nat.cast_zero
  /-
    case mk.mk'.intro
    α : Type u_3
    inst✝ : DivisionSemiring α
    a b : Nat
    hb : Ne (↑b) 0
    d : Nat
    h : Ne d 0
    n : Nat
    c : (↑n).natAbs.Coprime d
    hn : LE.le 0 { num := ↑n, den := d, den_nz := h, reduced := c }
    e : Eq (NNRat.divNat a b) ⟨{ num := ↑n, den := d, den_nz := h, reduced := c }, …
    hd : Ne (↑d) 0
    hb' : Ne b 0
    ⊢ Eq (↑⟨{ num := ↑n, den := d, den_nz := h, reduced := c }, hn⟩) (HDiv.hDiv ↑a …
  -/
  have hd' : d ≠ 0 := by rintro rfl; exact hd Nat.cast_zero
  /-
    case mk.mk'.intro
    α : Type u_3
    inst✝ : DivisionSemiring α
    a b : Nat
    hb : Ne (↑b) 0
    d : Nat
    h : Ne d 0
    n : Nat
    c : (↑n).natAbs.Coprime d
    hn : LE.le 0 { num := ↑n, den := d, den_nz := h, reduced := c }
    e : Eq (NNRat.divNat a b) ⟨{ num := ↑n, den := d, den_nz := h, reduced := c }, …
    hd : Ne (↑d) 0
    hb' : Ne b 0
    hd' : Ne d 0
    ⊢ Eq (↑⟨{ num := ↑n, den := d, den_nz := h, reduced := c }, hn⟩) (HDiv.hDiv ↑a …
  -/
  simp_rw [Rat.mk'_eq_divInt, mk_divInt, divNat_inj hb' hd'] at e
  /-
    case mk.mk'.intro
    α : Type u_3
    inst✝ : DivisionSemiring α
    a b : Nat
    hb : Ne (↑b) 0
    d : Nat
    h : Ne d 0
    n : Nat
    c : (↑n).natAbs.Coprime d
    hn : LE.le 0 { num := ↑n, den := d, den_nz := h, reduced := c }
    hd : Ne (↑d) 0
    hb' : Ne b 0
    hd' : Ne d 0
    e : Eq (HMul.hMul a d) (HMul.hMul n b)
    ⊢ Eq (↑⟨{ num := ↑n, den := d, den_nz := h, reduced := c }, hn⟩) (HDiv.hDiv ↑a …
  -/
  rw [cast_def]
  /-
    case mk.mk'.intro
    α : Type u_3
    inst✝ : DivisionSemiring α
    a b : Nat
    hb : Ne (↑b) 0
    d : Nat
    h : Ne d 0
    n : Nat
    c : (↑n).natAbs.Coprime d
    hn : LE.le 0 { num := ↑n, den := d, den_nz := h, reduced := c }
    hd : Ne (↑d) 0
    hb' : Ne b 0
    hd' : Ne d 0
    e : Eq (HMul.hMul a d) (HMul.hMul n b)
    ⊢ Eq (HDiv.hDiv ↑(NNRat.num ⟨{ num := ↑n, den := d, den_nz := h, reduced := c  …
  -/
  dsimp
  /-
    case mk.mk'.intro
    α : Type u_3
    inst✝ : DivisionSemiring α
    a b : Nat
    hb : Ne (↑b) 0
    d : Nat
    h : Ne d 0
    n : Nat
    c : (↑n).natAbs.Coprime d
    hn : LE.le 0 { num := ↑n, den := d, den_nz := h, reduced := c }
    hd : Ne (↑d) 0
    hb' : Ne b 0
    hd' : Ne d 0
    e : Eq (HMul.hMul a d) (HMul.hMul n b)
    ⊢ Eq (HDiv.hDiv ↑n ↑d) (HDiv.hDiv ↑a ↑b)
  -/
  rw [Commute.div_eq_div_iff _ hd hb]
    /-
      case mk.mk'.intro
      α : Type u_3
      inst✝ : DivisionSemiring α
      a b : Nat
      hb : Ne (↑b) 0
      d : Nat
      h : Ne d 0
      n : Nat
      c : (↑n).natAbs.Coprime d
      hn : LE.le 0 { num := ↑n, den := d, den_nz := h, reduced := c }
      hd : Ne (↑d) 0
      hb' : Ne b 0
      hd' : Ne d 0
      e : Eq (HMul.hMul a d) (HMul.hMul n b)
      ⊢ Eq (HMul.hMul ↑n ↑b) (HMul.hMul ↑a ↑d)
    -/
  · norm_cast
    /-
      case mk.mk'.intro
      α : Type u_3
      inst✝ : DivisionSemiring α
      a b : Nat
      hb : Ne (↑b) 0
      d : Nat
      h : Ne d 0
      n : Nat
      c : (↑n).natAbs.Coprime d
      hn : LE.le 0 { num := ↑n, den := d, den_nz := h, reduced := c }
      hd : Ne (↑d) 0
      hb' : Ne b 0
      hd' : Ne d 0
      e : Eq (HMul.hMul a d) (HMul.hMul n b)
      ⊢ Eq ↑(HMul.hMul n b) ↑(HMul.hMul a d)
    -/
    rw [e]
    /-
      🎉 no goals
    -/
  /-
    α : Type u_3
    inst✝ : DivisionSemiring α
    a b : Nat
    hb : Ne (↑b) 0
    d : Nat
    h : Ne d 0
    n : Nat
    c : (↑n).natAbs.Coprime d
    hn : LE.le 0 { num := ↑n, den := d, den_nz := h, reduced := c }
    hd : Ne (↑d) 0
    hb' : Ne b 0
    hd' : Ne d 0
    e : Eq (HMul.hMul a d) (HMul.hMul n b)
    ⊢ Commute ↑d ↑b
  -/
  exact b.commute_cast _
  /-
    🎉 no goals
  -/


@[norm_cast]
lemma cast_add_of_ne_zero (hq : (q.den : α) ≠ 0) (hr : (r.den : α) ≠ 0) :
    ↑(q + r) = (q + r : α) := by
  rw [add_def, cast_divNat_of_ne_zero, cast_def, cast_def, mul_comm _ q.den,
    (Nat.commute_cast _ _).div_add_div (Nat.commute_cast _ _) hq hr]
    /-
      α : Type u_3
      inst✝ : DivisionSemiring α
      q r : NNRat
      hq : Ne (↑q.den) 0
      hr : Ne (↑r.den) 0
      ⊢ Eq (HDiv.hDiv ↑(HAdd.hAdd (HMul.hMul q.num r.den) (HMul.hMul q.den r.num)) ↑ …
    -/
  · push_cast
    /-
      α : Type u_3
      inst✝ : DivisionSemiring α
      q r : NNRat
      hq : Ne (↑q.den) 0
      hr : Ne (↑r.den) 0
      ⊢ Eq (HDiv.hDiv (HAdd.hAdd (HMul.hMul ↑q.num ↑r.den) (HMul.hMul ↑q.den ↑r.num) …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case hb
      α : Type u_3
      inst✝ : DivisionSemiring α
      q r : NNRat
      hq : Ne (↑q.den) 0
      hr : Ne (↑r.den) 0
      ⊢ Ne (↑(HMul.hMul q.den r.den)) 0
    -/
  · push_cast
    /-
      case hb
      α : Type u_3
      inst✝ : DivisionSemiring α
      q r : NNRat
      hq : Ne (↑q.den) 0
      hr : Ne (↑r.den) 0
      ⊢ Ne (HMul.hMul ↑q.den ↑r.den) 0
    -/
    exact mul_ne_zero hq hr
    /-
      🎉 no goals
    -/


@[norm_cast]
lemma cast_mul_of_ne_zero (hq : (q.den : α) ≠ 0) (hr : (r.den : α) ≠ 0) :
    ↑(q * r) = (q * r : α) := by
  rw [mul_def, cast_divNat_of_ne_zero, cast_def, cast_def,
    (Nat.commute_cast _ _).div_mul_div_comm (Nat.commute_cast _ _)]
    /-
      α : Type u_3
      inst✝ : DivisionSemiring α
      q r : NNRat
      hq : Ne (↑q.den) 0
      hr : Ne (↑r.den) 0
      ⊢ Eq (HDiv.hDiv ↑(HMul.hMul q.num r.num) ↑(HMul.hMul q.den r.den)) (HDiv.hDiv  …
    -/
  · push_cast
    /-
      α : Type u_3
      inst✝ : DivisionSemiring α
      q r : NNRat
      hq : Ne (↑q.den) 0
      hr : Ne (↑r.den) 0
      ⊢ Eq (HDiv.hDiv (HMul.hMul ↑q.num ↑r.num) (HMul.hMul ↑q.den ↑r.den)) (HDiv.hDi …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case hb
      α : Type u_3
      inst✝ : DivisionSemiring α
      q r : NNRat
      hq : Ne (↑q.den) 0
      hr : Ne (↑r.den) 0
      ⊢ Ne (↑(HMul.hMul q.den r.den)) 0
    -/
  · push_cast
    /-
      case hb
      α : Type u_3
      inst✝ : DivisionSemiring α
      q r : NNRat
      hq : Ne (↑q.den) 0
      hr : Ne (↑r.den) 0
      ⊢ Ne (HMul.hMul ↑q.den ↑r.den) 0
    -/
    exact mul_ne_zero hq hr
    /-
      🎉 no goals
    -/


@[norm_cast]
lemma cast_inv_of_ne_zero (hq : (q.num : α) ≠ 0) : (q⁻¹ : ℚ≥0) = (q⁻¹ : α) := by
  /-
    α : Type u_3
    inst✝ : DivisionSemiring α
    q : NNRat
    hq : Ne (↑q.num) 0
    ⊢ Eq (↑(Inv.inv q)) (Inv.inv ↑q)
  -/
  rw [inv_def, cast_divNat_of_ne_zero _ hq, cast_def, inv_div]
  /-
    🎉 no goals
  -/


@[norm_cast]
lemma cast_div_of_ne_zero (hq : (q.den : α) ≠ 0) (hr : (r.num : α) ≠ 0) :
    ↑(q / r) = (q / r : α) := by
  rw [div_def, cast_divNat_of_ne_zero, cast_def, cast_def, div_eq_mul_inv (_ / _),
    inv_div, (Nat.commute_cast _ _).div_mul_div_comm (Nat.commute_cast _ _)]
    /-
      α : Type u_3
      inst✝ : DivisionSemiring α
      q r : NNRat
      hq : Ne (↑q.den) 0
      hr : Ne (↑r.num) 0
      ⊢ Eq (HDiv.hDiv ↑(HMul.hMul q.num r.den) ↑(HMul.hMul q.den r.num)) (HDiv.hDiv  …
    -/
  · push_cast
    /-
      α : Type u_3
      inst✝ : DivisionSemiring α
      q r : NNRat
      hq : Ne (↑q.den) 0
      hr : Ne (↑r.num) 0
      ⊢ Eq (HDiv.hDiv (HMul.hMul ↑q.num ↑r.den) (HMul.hMul ↑q.den ↑r.num)) (HDiv.hDi …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case hb
      α : Type u_3
      inst✝ : DivisionSemiring α
      q r : NNRat
      hq : Ne (↑q.den) 0
      hr : Ne (↑r.num) 0
      ⊢ Ne (↑(HMul.hMul q.den r.num)) 0
    -/
  · push_cast
    /-
      case hb
      α : Type u_3
      inst✝ : DivisionSemiring α
      q r : NNRat
      hq : Ne (↑q.den) 0
      hr : Ne (↑r.num) 0
      ⊢ Ne (HMul.hMul ↑q.den ↑r.num) 0
    -/
    exact mul_ne_zero hq hr
    /-
      🎉 no goals
    -/


@[simp, norm_cast]
theorem cast_intCast (n : ℤ) : ((n : ℚ) : α) = n :=
                                                      /-
                                                        α : Type u_3
                                                        inst✝ : DivisionRing α
                                                        n : Int
                                                        ⊢ Eq (HDiv.hDiv ↑n ↑1) ↑n
                                                      -/
  (cast_def _).trans <| show (n / (1 : ℕ) : α) = n by rw [Nat.cast_one, div_one]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp, norm_cast]
theorem cast_natCast (n : ℕ) : ((n : ℚ) : α) = n := by
  /-
    α : Type u_3
    inst✝ : DivisionRing α
    n : Nat
    ⊢ Eq ↑↑n ↑n
  -/
  rw [← Int.cast_natCast, cast_intCast, Int.cast_natCast]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-03-21")] alias cast_coe_int := cast_intCast

@[deprecated (since := "2024-03-21")] alias cast_coe_nat := cast_natCast


@[simp, norm_cast] lemma cast_ofNat (n : ℕ) [n.AtLeastTwo] :
    ((ofNat(n) : ℚ) : α) = (ofNat(n) : α) := by
  /-
    α : Type u_3
    inst✝¹ : DivisionRing α
    n : Nat
    inst✝ : n.AtLeastTwo
    ⊢ Eq (↑(OfNat.ofNat n)) (OfNat.ofNat n)
  -/
  simp [cast_def]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem cast_zero : ((0 : ℚ) : α) = 0 :=
  (cast_intCast _).trans Int.cast_zero


@[simp, norm_cast]
theorem cast_one : ((1 : ℚ) : α) = 1 :=
  (cast_intCast _).trans Int.cast_one


theorem cast_commute (r : ℚ) (a : α) : Commute (↑r) a := by
  /-
    α : Type u_3
    inst✝ : DivisionRing α
    r : Rat
    a : α
    ⊢ Commute (↑r) a
  -/
  simpa only [cast_def] using (r.1.cast_commute a).div_left (r.2.cast_commute a)
  /-
    🎉 no goals
  -/


theorem cast_comm (r : ℚ) (a : α) : (r : α) * a = a * r :=
  (cast_commute r a).eq


theorem commute_cast (a : α) (r : ℚ) : Commute a r :=
  (r.cast_commute a).symm


@[norm_cast]
lemma cast_divInt_of_ne_zero (a : ℤ) {b : ℤ} (b0 : (b : α) ≠ 0) : (a /. b : α) = a / b := by
  have b0' : b ≠ 0 := by
    refine mt ?_ b0
    simp +contextual
  /-
    α : Type u_3
    inst✝ : DivisionRing α
    a b : Int
    b0 : Ne (↑b) 0
    b0' : Ne b 0
    ⊢ Eq (↑(Rat.divInt a b)) (HDiv.hDiv ↑a ↑b)
  -/
  cases' e : a /. b with n d h c
  have d0 : (d : α) ≠ 0 := by
    intro d0
    have dd := den_dvd a b
    cases' show (d : ℤ) ∣ b by rwa [e] at dd with k ke
    have : (b : α) = (d : α) * (k : α) := by rw [ke, Int.cast_mul, Int.cast_natCast]
    rw [d0, zero_mul] at this
    contradiction
  /-
    case mk'
    α : Type u_3
    inst✝ : DivisionRing α
    a b : Int
    b0 : Ne (↑b) 0
    b0' : Ne b 0
    n : Int
    d : Nat
    h : Ne d 0
    c : n.natAbs.Coprime d
    e : Eq (Rat.divInt a b) { num := n, den := d, den_nz := h, reduced := c }
    d0 : Ne (↑d) 0
    ⊢ Eq (↑{ num := n, den := d, den_nz := h, reduced := c }) (HDiv.hDiv ↑a ↑b)
  -/
  rw [mk'_eq_divInt] at e
  have := congr_arg ((↑) : ℤ → α)
    ((divInt_eq_iff b0' <| ne_of_gt <| Int.natCast_pos.2 h.bot_lt).1 e)
  /-
    case mk'
    α : Type u_3
    inst✝ : DivisionRing α
    a b : Int
    b0 : Ne (↑b) 0
    b0' : Ne b 0
    n : Int
    d : Nat
    h : Ne d 0
    c : n.natAbs.Coprime d
    e : Eq (Rat.divInt a b) (Rat.divInt n ↑d)
    d0 : Ne (↑d) 0
    this : Eq ↑(HMul.hMul a ↑d) ↑(HMul.hMul n b)
    ⊢ Eq (↑{ num := n, den := d, den_nz := h, reduced := c }) (HDiv.hDiv ↑a ↑b)
  -/
  rw [Int.cast_mul, Int.cast_mul, Int.cast_natCast] at this
  rw [eq_comm, cast_def, div_eq_mul_inv, eq_div_iff_mul_eq d0, mul_assoc, (d.commute_cast _).eq,
    ← mul_assoc, this, mul_assoc, mul_inv_cancel₀ b0, mul_one]


@[norm_cast]
lemma cast_mkRat_of_ne_zero (a : ℤ) {b : ℕ} (hb : (b : α) ≠ 0) : (mkRat a b : α) = a / b := by
  /-
    α : Type u_3
    inst✝ : DivisionRing α
    a : Int
    b : Nat
    hb : Ne (↑b) 0
    ⊢ Eq (↑(mkRat a b)) (HDiv.hDiv ↑a ↑b)
  -/
  rw [Rat.mkRat_eq_divInt, cast_divInt_of_ne_zero, Int.cast_natCast]; rwa [Int.cast_natCast]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[norm_cast]
lemma cast_add_of_ne_zero {q r : ℚ} (hq : (q.den : α) ≠ 0) (hr : (r.den : α) ≠ 0) :
    (q + r : ℚ) = (q + r : α) := by
  rw [add_def', cast_mkRat_of_ne_zero, cast_def, cast_def, mul_comm r.num,
    (Nat.cast_commute _ _).div_add_div (Nat.commute_cast _ _) hq hr]
    /-
      α : Type u_3
      inst✝ : DivisionRing α
      q r : Rat
      hq : Ne (↑q.den) 0
      hr : Ne (↑r.den) 0
      ⊢ Eq (HDiv.hDiv ↑(HAdd.hAdd (HMul.hMul q.num ↑r.den) (HMul.hMul (↑q.den) r.num …
    -/
  · push_cast
    /-
      α : Type u_3
      inst✝ : DivisionRing α
      q r : Rat
      hq : Ne (↑q.den) 0
      hr : Ne (↑r.den) 0
      ⊢ Eq (HDiv.hDiv (HAdd.hAdd (HMul.hMul ↑q.num ↑r.den) (HMul.hMul ↑q.den ↑r.num) …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case hb
      α : Type u_3
      inst✝ : DivisionRing α
      q r : Rat
      hq : Ne (↑q.den) 0
      hr : Ne (↑r.den) 0
      ⊢ Ne (↑(HMul.hMul q.den r.den)) 0
    -/
  · push_cast
    /-
      case hb
      α : Type u_3
      inst✝ : DivisionRing α
      q r : Rat
      hq : Ne (↑q.den) 0
      hr : Ne (↑r.den) 0
      ⊢ Ne (HMul.hMul ↑q.den ↑r.den) 0
    -/
    exact mul_ne_zero hq hr
    /-
      🎉 no goals
    -/


                                                                   /-
                                                                     α : Type u_3
                                                                     inst✝ : DivisionRing α
                                                                     q : Rat
                                                                     ⊢ Eq (↑(Neg.neg q)) (Neg.neg ↑q)
                                                                   -/
@[simp, norm_cast] lemma cast_neg (q : ℚ) : ↑(-q) = (-q : α) := by simp [cast_def, neg_div]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[norm_cast] lemma cast_sub_of_ne_zero (hp : (p.den : α) ≠ 0) (hq : (q.den : α) ≠ 0) :
                                 /-
                                   α : Type u_3
                                   inst✝ : DivisionRing α
                                   p q : Rat
                                   hp : Ne (↑p.den) 0
                                   hq : Ne (↑q.den) 0
                                   ⊢ Eq (↑(HSub.hSub p q)) (HSub.hSub ↑p ↑q)
                                 -/
    ↑(p - q) = (p - q : α) := by simp [sub_eq_add_neg, cast_add_of_ne_zero, hp, hq]
                                 /-
                                   🎉 no goals
                                 -/


@[norm_cast] lemma cast_mul_of_ne_zero (hp : (p.den : α) ≠ 0) (hq : (q.den : α) ≠ 0) :
    ↑(p * q) = (p * q : α) := by
  rw [mul_eq_mkRat, cast_mkRat_of_ne_zero, cast_def, cast_def,
    (Nat.commute_cast _ _).div_mul_div_comm (Int.commute_cast _ _)]
    /-
      α : Type u_3
      inst✝ : DivisionRing α
      p q : Rat
      hp : Ne (↑p.den) 0
      hq : Ne (↑q.den) 0
      ⊢ Eq (HDiv.hDiv ↑(HMul.hMul p.num q.num) ↑(HMul.hMul p.den q.den)) (HDiv.hDiv  …
    -/
  · push_cast
    /-
      α : Type u_3
      inst✝ : DivisionRing α
      p q : Rat
      hp : Ne (↑p.den) 0
      hq : Ne (↑q.den) 0
      ⊢ Eq (HDiv.hDiv (HMul.hMul ↑p.num ↑q.num) (HMul.hMul ↑p.den ↑q.den)) (HDiv.hDi …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case hb
      α : Type u_3
      inst✝ : DivisionRing α
      p q : Rat
      hp : Ne (↑p.den) 0
      hq : Ne (↑q.den) 0
      ⊢ Ne (↑(HMul.hMul p.den q.den)) 0
    -/
  · push_cast
    /-
      case hb
      α : Type u_3
      inst✝ : DivisionRing α
      p q : Rat
      hp : Ne (↑p.den) 0
      hq : Ne (↑q.den) 0
      ⊢ Ne (HMul.hMul ↑p.den ↑q.den) 0
    -/
    exact mul_ne_zero hp hq
    /-
      🎉 no goals
    -/


@[norm_cast]
lemma cast_inv_of_ne_zero (hq : (q.num : α) ≠ 0) : ↑(q⁻¹) = (q⁻¹ : α) := by
  /-
    α : Type u_3
    inst✝ : DivisionRing α
    q : Rat
    hq : Ne (↑q.num) 0
    ⊢ Eq (↑(Inv.inv q)) (Inv.inv ↑q)
  -/
  rw [inv_def', cast_divInt_of_ne_zero _ hq, cast_def, inv_div, Int.cast_natCast]
  /-
    🎉 no goals
  -/


@[norm_cast] lemma cast_div_of_ne_zero (hp : (p.den : α) ≠ 0) (hq : (q.num : α) ≠ 0) :
    ↑(p / q) = (p / q : α) := by
  rw [div_def', cast_divInt_of_ne_zero, cast_def, cast_def, div_eq_mul_inv (_ / _), inv_div,
    (Int.commute_cast _ _).div_mul_div_comm (Nat.commute_cast _ _)]
    /-
      α : Type u_3
      inst✝ : DivisionRing α
      p q : Rat
      hp : Ne (↑p.den) 0
      hq : Ne (↑q.num) 0
      ⊢ Eq (HDiv.hDiv ↑(HMul.hMul p.num ↑q.den) ↑(HMul.hMul (↑p.den) q.num)) (HDiv.h …
    -/
  · push_cast
    /-
      α : Type u_3
      inst✝ : DivisionRing α
      p q : Rat
      hp : Ne (↑p.den) 0
      hq : Ne (↑q.num) 0
      ⊢ Eq (HDiv.hDiv (HMul.hMul ↑p.num ↑q.den) (HMul.hMul ↑p.den ↑q.num)) (HDiv.hDi …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case b0
      α : Type u_3
      inst✝ : DivisionRing α
      p q : Rat
      hp : Ne (↑p.den) 0
      hq : Ne (↑q.num) 0
      ⊢ Ne (↑(HMul.hMul (↑p.den) q.num)) 0
    -/
  · push_cast
    /-
      case b0
      α : Type u_3
      inst✝ : DivisionRing α
      p q : Rat
      hp : Ne (↑p.den) 0
      hq : Ne (↑q.num) 0
      ⊢ Ne (HMul.hMul ↑p.den ↑q.num) 0
    -/
    exact mul_ne_zero hp hq
    /-
      🎉 no goals
    -/


@[simp] lemma map_nnratCast [DivisionSemiring α] [DivisionSemiring β] [RingHomClass F α β] (f : F)
                              /-
                                F : Type u_1
                                α : Type u_3
                                β : Type u_4
                                inst✝³ : FunLike F α β
                                inst✝² : DivisionSemiring α
                                inst✝¹ : DivisionSemiring β
                                inst✝ : RingHomClass F α β
                                f : F
                                q : NNRat
                                ⊢ Eq (f ↑q) ↑q
                              -/
    (q : ℚ≥0) : f q = q := by simp_rw [NNRat.cast_def, map_div₀, map_natCast]
                              /-
                                🎉 no goals
                              -/


@[simp]
lemma eq_nnratCast [DivisionSemiring α] [FunLike F ℚ≥0 α] [RingHomClass F ℚ≥0 α] (f : F) (q : ℚ≥0) :
                  /-
                    F : Type u_1
                    α : Type u_3
                    inst✝² : DivisionSemiring α
                    inst✝¹ : FunLike F NNRat α
                    inst✝ : RingHomClass F NNRat α
                    f : F
                    q : NNRat
                    ⊢ Eq (f q) ↑q
                  -/
    f q = q := by rw [← map_nnratCast f, NNRat.cast_id]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem map_ratCast [DivisionRing α] [DivisionRing β] [RingHomClass F α β] (f : F) (q : ℚ) :
                  /-
                    F : Type u_1
                    α : Type u_3
                    β : Type u_4
                    inst✝³ : FunLike F α β
                    inst✝² : DivisionRing α
                    inst✝¹ : DivisionRing β
                    inst✝ : RingHomClass F α β
                    f : F
                    q : Rat
                    ⊢ Eq (f ↑q) ↑q
                  -/
    f q = q := by rw [cast_def, map_div₀, map_intCast, map_natCast, cast_def]
                  /-
                    🎉 no goals
                  -/


@[simp] lemma eq_ratCast [DivisionRing α] [FunLike F ℚ α] [RingHomClass F ℚ α] (f : F) (q : ℚ) :
                  /-
                    F : Type u_1
                    α : Type u_3
                    inst✝² : DivisionRing α
                    inst✝¹ : FunLike F Rat α
                    inst✝ : RingHomClass F Rat α
                    f : F
                    q : Rat
                    ⊢ Eq (f q) ↑q
                  -/
    f q = q := by rw [← map_ratCast f, Rat.cast_id]
                  /-
                    🎉 no goals
                  -/


/-- If monoid with zero homs `f` and `g` from `ℚ≥0` agree on the naturals then they are equal. -/
lemma ext_nnrat' (h : ∀ n : ℕ, f n = g n) : f = g :=
  (DFunLike.ext f g) fun r => by
    /-
      F : Type u_1
      M₀ : Type u_5
      inst✝² : MonoidWithZero M₀
      inst✝¹ : FunLike F NNRat M₀
      inst✝ : MonoidWithZeroHomClass F NNRat M₀
      f g : F
      h : ∀ (n : Nat), Eq (f ↑n) (g ↑n)
      r : NNRat
      ⊢ Eq (f r) (g r)
    -/
    rw [← r.num_div_den, div_eq_mul_inv, map_mul, map_mul, h, eq_on_inv₀ f g]
    /-
      F : Type u_1
      M₀ : Type u_5
      inst✝² : MonoidWithZero M₀
      inst✝¹ : FunLike F NNRat M₀
      inst✝ : MonoidWithZeroHomClass F NNRat M₀
      f g : F
      h : ∀ (n : Nat), Eq (f ↑n) (g ↑n)
      r : NNRat
      ⊢ Eq (f ↑r.den) (g ↑r.den)
    -/
    apply h
    /-
      🎉 no goals
    -/


/-- If monoid with zero homs `f` and `g` from `ℚ≥0` agree on the naturals then they are equal.

See note [partially-applied ext lemmas] for why `comp` is used here. -/
@[ext]
lemma ext_nnrat {f g : ℚ≥0 →*₀ M₀}
    (h : f.comp (Nat.castRingHom ℚ≥0 : ℕ →*₀ ℚ≥0) = g.comp (Nat.castRingHom ℚ≥0)) : f = g :=
  ext_nnrat' <| DFunLike.congr_fun h


/-- If monoid with zero homs `f` and `g` from `ℚ≥0` agree on the positive naturals then they are
equal. -/
lemma ext_nnrat_on_pnat (same_on_pnat : ∀ n : ℕ, 0 < n → f n = g n) : f = g :=
  ext_nnrat' <| DFunLike.congr_fun <| ext_nat''
    ((f : ℚ≥0 →*₀ M₀).comp (Nat.castRingHom ℚ≥0 : ℕ →*₀ ℚ≥0))
                                                                  /-
                                                                    F : Type u_1
                                                                    M₀ : Type u_5
                                                                    inst✝² : MonoidWithZero M₀
                                                                    inst✝¹ : FunLike F NNRat M₀
                                                                    inst✝ : MonoidWithZeroHomClass F NNRat M₀
                                                                    f g : F
                                                                    same_on_pnat : ∀ (n : Nat), LT.lt 0 n → Eq (f ↑n) (g ↑n)
                                                                    ⊢ ∀ {n : Nat}, LT.lt 0 n → Eq (((↑f).comp ↑(Nat.castRingHom NNRat)) n) (((↑g). …
                                                                  -/
    ((g : ℚ≥0 →*₀ M₀).comp (Nat.castRingHom ℚ≥0 : ℕ →*₀ ℚ≥0)) (by simpa)
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- If monoid with zero homs `f` and `g` from `ℚ` agree on the integers then they are equal. -/
theorem ext_rat' (h : ∀ m : ℤ, f m = g m) : f = g :=
  (DFunLike.ext f g) fun r => by
    rw [← r.num_div_den, div_eq_mul_inv, map_mul, map_mul, h, ← Int.cast_natCast,
      eq_on_inv₀ f g]
    /-
      F : Type u_1
      M₀ : Type u_5
      inst✝² : MonoidWithZero M₀
      inst✝¹ : FunLike F Rat M₀
      inst✝ : MonoidWithZeroHomClass F Rat M₀
      f g : F
      h : ∀ (m : Int), Eq (f ↑m) (g ↑m)
      r : Rat
      ⊢ Eq (f ↑↑r.den) (g ↑↑r.den)
    -/
    apply h
    /-
      🎉 no goals
    -/


/-- If monoid with zero homs `f` and `g` from `ℚ` agree on the integers then they are equal.

See note [partially-applied ext lemmas] for why `comp` is used here. -/
@[ext]
theorem ext_rat {f g : ℚ →*₀ M₀}
    (h : f.comp (Int.castRingHom ℚ : ℤ →*₀ ℚ) = g.comp (Int.castRingHom ℚ)) : f = g :=
  ext_rat' <| DFunLike.congr_fun h


/-- If monoid with zero homs `f` and `g` from `ℚ` agree on the positive naturals and `-1` then
they are equal. -/
theorem ext_rat_on_pnat (same_on_neg_one : f (-1) = g (-1))
    (same_on_pnat : ∀ n : ℕ, 0 < n → f n = g n) : f = g :=
  ext_rat' <|
    DFunLike.congr_fun <|
      show
        (f : ℚ →*₀ M₀).comp (Int.castRingHom ℚ : ℤ →*₀ ℚ) =
          (g : ℚ →*₀ M₀).comp (Int.castRingHom ℚ : ℤ →*₀ ℚ)
                          /-
                            F : Type u_1
                            M₀ : Type u_5
                            inst✝² : MonoidWithZero M₀
                            inst✝¹ : FunLike F Rat M₀
                            inst✝ : MonoidWithZeroHomClass F Rat M₀
                            f g : F
                            same_on_neg_one : Eq (f (-1)) (g (-1))
                            same_on_pnat : ∀ (n : Nat), LT.lt 0 n → Eq (f ↑n) (g ↑n)
                            ⊢ Eq (((↑f).comp ↑(Int.castRingHom Rat)) (-1)) (((↑g).comp ↑(Int.castRingHom R …
                          -/
                          /-
                            🎉 no goals
                          -/
        from ext_int' (by simpa) (by simpa)
                                     /-
                                       🎉 no goals
                                     -/


/-- Any two ring homomorphisms from `ℚ` to a semiring are equal. If the codomain is a division ring,
then this lemma follows from `eq_ratCast`. -/
theorem RingHom.ext_rat {R : Type*} [Semiring R] [FunLike F ℚ R] [RingHomClass F ℚ R] (f g : F) :
    f = g :=
  MonoidWithZeroHomClass.ext_rat' <|
    RingHom.congr_fun <|
      ((f : ℚ →+* R).comp (Int.castRingHom ℚ)).ext_int ((g : ℚ →+* R).comp (Int.castRingHom ℚ))


instance NNRat.subsingleton_ringHom {R : Type*} [Semiring R] : Subsingleton (ℚ≥0 →+* R) where
                                                       /-
                                                         F : Type u_1
                                                         ι : Type u_2
                                                         α : Type u_3
                                                         β : Type u_4
                                                         inst✝¹ : FunLike F α β
                                                         R : Type u_5
                                                         inst✝ : Semiring R
                                                         f g : RingHom NNRat R
                                                         ⊢ ∀ (n : Nat), Eq (f ↑n) (g ↑n)
                                                       -/
  allEq f g := MonoidWithZeroHomClass.ext_nnrat' <| by simp
                                                       /-
                                                         🎉 no goals
                                                       -/


instance Rat.subsingleton_ringHom {R : Type*} [Semiring R] : Subsingleton (ℚ →+* R) :=
  ⟨RingHom.ext_rat⟩


instance (priority := 100) instDistribSMul : DistribSMul ℚ≥0 α where
                    /-
                      F : Type u_1
                      ι : Type u_2
                      α : Type u_3
                      β : Type u_4
                      inst✝¹ : FunLike F α β
                      inst✝ : DivisionSemiring α
                      a : NNRat
                      ⊢ Eq (HSMul.hSMul a 0) 0
                    -/
  smul_zero a := by rw [smul_def, mul_zero]
                    /-
                      🎉 no goals
                    -/
                       /-
                         F : Type u_1
                         ι : Type u_2
                         α : Type u_3
                         β : Type u_4
                         inst✝¹ : FunLike F α β
                         inst✝ : DivisionSemiring α
                         a : NNRat
                         x y : α
                         ⊢ Eq (HSMul.hSMul a (HAdd.hAdd x y)) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul …
                       -/
  smul_add a x y := by rw [smul_def, smul_def, smul_def, mul_add]
                       /-
                         🎉 no goals
                       -/


instance instIsScalarTowerRight : IsScalarTower ℚ≥0 α α where
                         /-
                           F : Type u_1
                           ι : Type u_2
                           α : Type u_3
                           β : Type u_4
                           inst✝¹ : FunLike F α β
                           inst✝ : DivisionSemiring α
                           a : NNRat
                           x y : α
                           ⊢ Eq (HSMul.hSMul (HSMul.hSMul a x) y) (HSMul.hSMul a (HSMul.hSMul x y))
                         -/
  smul_assoc a x y := by simp only [smul_def, smul_eq_mul, mul_assoc]
                         /-
                           🎉 no goals
                         -/


instance (priority := 100) instDistribSMul : DistribSMul ℚ α where
                    /-
                      F : Type u_1
                      ι : Type u_2
                      α : Type u_3
                      β : Type u_4
                      inst✝¹ : FunLike F α β
                      inst✝ : DivisionRing α
                      a : Rat
                      ⊢ Eq (HSMul.hSMul a 0) 0
                    -/
  smul_zero a := by rw [smul_def, mul_zero]
                    /-
                      🎉 no goals
                    -/
                       /-
                         F : Type u_1
                         ι : Type u_2
                         α : Type u_3
                         β : Type u_4
                         inst✝¹ : FunLike F α β
                         inst✝ : DivisionRing α
                         a : Rat
                         x y : α
                         ⊢ Eq (HSMul.hSMul a (HAdd.hAdd x y)) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul …
                       -/
  smul_add a x y := by rw [smul_def, smul_def, smul_def, mul_add]
                       /-
                         🎉 no goals
                       -/


instance instIsScalarTowerRight : IsScalarTower ℚ α α where
                         /-
                           F : Type u_1
                           ι : Type u_2
                           α : Type u_3
                           β : Type u_4
                           inst✝¹ : FunLike F α β
                           inst✝ : DivisionRing α
                           a : Rat
                           x y : α
                           ⊢ Eq (HSMul.hSMul (HSMul.hSMul a x) y) (HSMul.hSMul a (HSMul.hSMul x y))
                         -/
  smul_assoc a x y := by simp only [smul_def, smul_eq_mul, mul_assoc]
                         /-
                           🎉 no goals
                         -/


