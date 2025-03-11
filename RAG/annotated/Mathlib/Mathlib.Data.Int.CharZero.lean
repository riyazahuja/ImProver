@[simp, norm_cast]
theorem cast_div_charZero {k : Type*} [DivisionRing k] [CharZero k] {m n : ℤ} (n_dvd : n ∣ m) :
    ((m / n : ℤ) : k) = m / n := by
  /-
    k : Type u_3
    inst✝¹ : DivisionRing k
    inst✝ : CharZero k
    m n : Int
    n_dvd : Dvd.dvd n m
    ⊢ Eq (↑(HDiv.hDiv m n)) (HDiv.hDiv ↑m ↑n)
  -/
  rcases eq_or_ne n 0 with (rfl | hn)
    /-
      case inl
      k : Type u_3
      inst✝¹ : DivisionRing k
      inst✝ : CharZero k
      m : Int
      n_dvd : Dvd.dvd 0 m
      ⊢ Eq (↑(HDiv.hDiv m 0)) (HDiv.hDiv ↑m ↑0)
    -/
  · simp [Int.ediv_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr
      k : Type u_3
      inst✝¹ : DivisionRing k
      inst✝ : CharZero k
      m n : Int
      n_dvd : Dvd.dvd n m
      hn : Ne n 0
      ⊢ Eq (↑(HDiv.hDiv m n)) (HDiv.hDiv ↑m ↑n)
    -/
  · exact cast_div n_dvd (cast_ne_zero.mpr hn)
    /-
      🎉 no goals
    -/

-- Necessary for confluence with `ofNat_ediv` and `cast_div_charZero`.

@[simp, norm_cast]
theorem cast_div_ofNat_charZero {k : Type*} [DivisionRing k] [CharZero k] {m n : ℕ}
    (n_dvd : n ∣ m) : (((m : ℤ) / (n : ℤ) : ℤ) : k) = m / n := by
  /-
    k : Type u_3
    inst✝¹ : DivisionRing k
    inst✝ : CharZero k
    m n : Nat
    n_dvd : Dvd.dvd n m
    ⊢ Eq (↑(HDiv.hDiv ↑m ↑n)) (HDiv.hDiv ↑m ↑n)
  -/
  rw [cast_div_charZero (Int.ofNat_dvd.mpr n_dvd), cast_natCast, cast_natCast]
  /-
    🎉 no goals
  -/


theorem RingHom.injective_int {α : Type*} [NonAssocRing α] (f : ℤ →+* α) [CharZero α] :
    Function.Injective f :=
  Subsingleton.elim (Int.castRingHom _) f ▸ Int.cast_injective


lemma support_intCast (hn : n ≠ 0) : support (n : α → β) = univ :=
  support_const <| Int.cast_ne_zero.2 hn


@[deprecated (since := "2024-04-17")]
alias support_int_cast := support_intCast


lemma mulSupport_intCast (hn : n ≠ 1) : mulSupport (n : α → β) = univ :=
  mulSupport_const <| Int.cast_ne_one.2 hn


@[deprecated (since := "2024-04-17")]
alias mulSupport_int_cast := mulSupport_intCast


