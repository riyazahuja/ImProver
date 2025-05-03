/-- Shows that the fractional parts of the stream are in `[0,1)`. -/
theorem nth_stream_fr_nonneg_lt_one {ifp_n : IntFractPair K}
    (nth_stream_eq : IntFractPair.stream v n = some ifp_n) : 0 ≤ ifp_n.fr ∧ ifp_n.fr < 1 := by
  cases n with
  | zero =>
    have : IntFractPair.of v = ifp_n := by injection nth_stream_eq
    rw [← this, IntFractPair.of]
    exact ⟨fract_nonneg _, fract_lt_one _⟩
  | succ =>
    rcases succ_nth_stream_eq_some_iff.1 nth_stream_eq with ⟨_, _, _, ifp_of_eq_ifp_n⟩
    rw [← ifp_of_eq_ifp_n, IntFractPair.of]
    exact ⟨fract_nonneg _, fract_lt_one _⟩


/-- Shows that the fractional parts of the stream are nonnegative. -/
theorem nth_stream_fr_nonneg {ifp_n : IntFractPair K}
    (nth_stream_eq : IntFractPair.stream v n = some ifp_n) : 0 ≤ ifp_n.fr :=
  (nth_stream_fr_nonneg_lt_one nth_stream_eq).left


/-- Shows that the fractional parts of the stream are smaller than one. -/
theorem nth_stream_fr_lt_one {ifp_n : IntFractPair K}
    (nth_stream_eq : IntFractPair.stream v n = some ifp_n) : ifp_n.fr < 1 :=
  (nth_stream_fr_nonneg_lt_one nth_stream_eq).right


/-- Shows that the integer parts of the stream are at least one. -/
theorem one_le_succ_nth_stream_b {ifp_succ_n : IntFractPair K}
    (succ_nth_stream_eq : IntFractPair.stream v (n + 1) = some ifp_succ_n) : 1 ≤ ifp_succ_n.b := by
  obtain ⟨ifp_n, nth_stream_eq, stream_nth_fr_ne_zero, ⟨-⟩⟩ :
      ∃ ifp_n, IntFractPair.stream v n = some ifp_n ∧ ifp_n.fr ≠ 0
        ∧ IntFractPair.of ifp_n.fr⁻¹ = ifp_succ_n :=
    succ_nth_stream_eq_some_iff.1 succ_nth_stream_eq
  rw [IntFractPair.of, le_floor, cast_one, one_le_inv₀
    ((nth_stream_fr_nonneg nth_stream_eq).lt_of_ne' stream_nth_fr_ne_zero)]
  /-
    case intro.intro.intro.refl
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    ifp_n : GenContFract.IntFractPair K
    nth_stream_eq : Eq (GenContFract.IntFractPair.stream v n) (Option.some ifp_n)
    stream_nth_fr_ne_zero : Ne ifp_n.fr 0
    succ_nth_stream_eq : Eq (GenContFract.IntFractPair.stream v (HAdd.hAdd n 1)) ( …
    ⊢ LE.le ifp_n.fr 1
  -/
  exact (nth_stream_fr_lt_one nth_stream_eq).le
  /-
    🎉 no goals
  -/


/--
Shows that the `n + 1`th integer part `bₙ₊₁` of the stream is smaller or equal than the inverse of
the `n`th fractional part `frₙ` of the stream.
This result is straight-forward as `bₙ₊₁` is defined as the floor of `1 / frₙ`.
-/
theorem succ_nth_stream_b_le_nth_stream_fr_inv {ifp_n ifp_succ_n : IntFractPair K}
    (nth_stream_eq : IntFractPair.stream v n = some ifp_n)
    (succ_nth_stream_eq : IntFractPair.stream v (n + 1) = some ifp_succ_n) :
    (ifp_succ_n.b : K) ≤ ifp_n.fr⁻¹ := by
  suffices (⌊ifp_n.fr⁻¹⌋ : K) ≤ ifp_n.fr⁻¹ by
    obtain ⟨_, ifp_n_fr⟩ := ifp_n
    have : ifp_n_fr ≠ 0 := by
      intro h
      simp [h, IntFractPair.stream, nth_stream_eq] at succ_nth_stream_eq
    have : IntFractPair.of ifp_n_fr⁻¹ = ifp_succ_n := by
      simpa [this, IntFractPair.stream, nth_stream_eq, Option.coe_def] using succ_nth_stream_eq
    rwa [← this]
  /-
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    ifp_n ifp_succ_n : GenContFract.IntFractPair K
    nth_stream_eq : Eq (GenContFract.IntFractPair.stream v n) (Option.some ifp_n)
    succ_nth_stream_eq : Eq (GenContFract.IntFractPair.stream v (HAdd.hAdd n 1)) ( …
    ⊢ LE.le (↑(Int.floor (Inv.inv ifp_n.fr))) (Inv.inv ifp_n.fr)
  -/
  exact floor_le ifp_n.fr⁻¹
  /-
    🎉 no goals
  -/


/-- Shows that the integer parts of the continued fraction are at least one. -/
theorem of_one_le_get?_partDen {b : K}
    (nth_partDen_eq : (of v).partDens.get? n = some b) : 1 ≤ b := by
  obtain ⟨gp_n, nth_s_eq, ⟨-⟩⟩ : ∃ gp_n, (of v).s.get? n = some gp_n ∧ gp_n.b = b :=
    exists_s_b_of_partDen nth_partDen_eq
  obtain ⟨ifp_n, succ_nth_stream_eq, ifp_n_b_eq_gp_n_b⟩ :
      ∃ ifp, IntFractPair.stream v (n + 1) = some ifp ∧ (ifp.b : K) = gp_n.b :=
    IntFractPair.exists_succ_get?_stream_of_gcf_of_get?_eq_some nth_s_eq
  /-
    case intro.intro.refl.intro.intro
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    gp_n : GenContFract.Pair K
    nth_s_eq : Eq ((GenContFract.of v).s.get? n) (Option.some gp_n)
    nth_partDen_eq : Eq ((GenContFract.of v).partDens.get? n) (Option.some gp_n.b)
    ifp_n : GenContFract.IntFractPair K
    succ_nth_stream_eq : Eq (GenContFract.IntFractPair.stream v (HAdd.hAdd n 1)) ( …
    ifp_n_b_eq_gp_n_b : Eq (↑ifp_n.b) gp_n.b
    ⊢ LE.le 1 gp_n.b
  -/
  rw [← ifp_n_b_eq_gp_n_b]
  /-
    case intro.intro.refl.intro.intro
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    gp_n : GenContFract.Pair K
    nth_s_eq : Eq ((GenContFract.of v).s.get? n) (Option.some gp_n)
    nth_partDen_eq : Eq ((GenContFract.of v).partDens.get? n) (Option.some gp_n.b)
    ifp_n : GenContFract.IntFractPair K
    succ_nth_stream_eq : Eq (GenContFract.IntFractPair.stream v (HAdd.hAdd n 1)) ( …
    ifp_n_b_eq_gp_n_b : Eq (↑ifp_n.b) gp_n.b
    ⊢ LE.le 1 ↑ifp_n.b
  -/
  exact mod_cast IntFractPair.one_le_succ_nth_stream_b succ_nth_stream_eq
  /-
    🎉 no goals
  -/


/--
Shows that the partial numerators `aᵢ` of the continued fraction are equal to one and the partial
denominators `bᵢ` correspond to integers.
-/
theorem of_partNum_eq_one_and_exists_int_partDen_eq {gp : GenContFract.Pair K}
    (nth_s_eq : (of v).s.get? n = some gp) : gp.a = 1 ∧ ∃ z : ℤ, gp.b = (z : K) := by
  obtain ⟨ifp, stream_succ_nth_eq, -⟩ : ∃ ifp, IntFractPair.stream v (n + 1) = some ifp ∧ _ :=
    IntFractPair.exists_succ_get?_stream_of_gcf_of_get?_eq_some nth_s_eq
  have : gp = ⟨1, ifp.b⟩ := by
    have : (of v).s.get? n = some ⟨1, ifp.b⟩ :=
      get?_of_eq_some_of_succ_get?_intFractPair_stream stream_succ_nth_eq
    have : some gp = some ⟨1, ifp.b⟩ := by rwa [nth_s_eq] at this
    injection this
  /-
    case intro.intro
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    gp : GenContFract.Pair K
    nth_s_eq : Eq ((GenContFract.of v).s.get? n) (Option.some gp)
    ifp : GenContFract.IntFractPair K
    stream_succ_nth_eq : Eq (GenContFract.IntFractPair.stream v (HAdd.hAdd n 1)) ( …
    this : Eq gp { a := 1, b := ↑ifp.b }
    ⊢ And (Eq gp.a 1) (Exists fun z => Eq gp.b ↑z)
  -/
  simp [this]
  /-
    🎉 no goals
  -/


/-- Shows that the partial numerators `aᵢ` are equal to one. -/
theorem of_partNum_eq_one {a : K} (nth_partNum_eq : (of v).partNums.get? n = some a) :
    a = 1 := by
  obtain ⟨gp, nth_s_eq, gp_a_eq_a_n⟩ : ∃ gp, (of v).s.get? n = some gp ∧ gp.a = a :=
    exists_s_a_of_partNum nth_partNum_eq
  /-
    case intro.intro
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    a : K
    nth_partNum_eq : Eq ((GenContFract.of v).partNums.get? n) (Option.some a)
    gp : GenContFract.Pair K
    nth_s_eq : Eq ((GenContFract.of v).s.get? n) (Option.some gp)
    gp_a_eq_a_n : Eq gp.a a
    ⊢ Eq a 1
  -/
  have : gp.a = 1 := (of_partNum_eq_one_and_exists_int_partDen_eq nth_s_eq).left
  /-
    case intro.intro
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    a : K
    nth_partNum_eq : Eq ((GenContFract.of v).partNums.get? n) (Option.some a)
    gp : GenContFract.Pair K
    nth_s_eq : Eq ((GenContFract.of v).s.get? n) (Option.some gp)
    gp_a_eq_a_n : Eq gp.a a
    this : Eq gp.a 1
    ⊢ Eq a 1
  -/
  rwa [gp_a_eq_a_n] at this
  /-
    🎉 no goals
  -/


/-- Shows that the partial denominators `bᵢ` correspond to an integer. -/
theorem exists_int_eq_of_partDen {b : K}
    (nth_partDen_eq : (of v).partDens.get? n = some b) : ∃ z : ℤ, b = (z : K) := by
  obtain ⟨gp, nth_s_eq, gp_b_eq_b_n⟩ : ∃ gp, (of v).s.get? n = some gp ∧ gp.b = b :=
    exists_s_b_of_partDen nth_partDen_eq
  /-
    case intro.intro
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    b : K
    nth_partDen_eq : Eq ((GenContFract.of v).partDens.get? n) (Option.some b)
    gp : GenContFract.Pair K
    nth_s_eq : Eq ((GenContFract.of v).s.get? n) (Option.some gp)
    gp_b_eq_b_n : Eq gp.b b
    ⊢ Exists fun z => Eq b ↑z
  -/
  have : ∃ z : ℤ, gp.b = (z : K) := (of_partNum_eq_one_and_exists_int_partDen_eq nth_s_eq).right
  /-
    case intro.intro
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    b : K
    nth_partDen_eq : Eq ((GenContFract.of v).partDens.get? n) (Option.some b)
    gp : GenContFract.Pair K
    nth_s_eq : Eq ((GenContFract.of v).s.get? n) (Option.some gp)
    gp_b_eq_b_n : Eq gp.b b
    this : Exists fun z => Eq gp.b ↑z
    ⊢ Exists fun z => Eq b ↑z
  -/
  rwa [gp_b_eq_b_n] at this
  /-
    🎉 no goals
  -/


theorem GenContFract.of_isSimpContFract :
    (of v).IsSimpContFract := fun _ _ nth_partNum_eq =>
  of_partNum_eq_one nth_partNum_eq


/-- Creates the simple continued fraction of a value. -/
nonrec def SimpContFract.of : SimpContFract K :=
  ⟨of v, GenContFract.of_isSimpContFract v⟩


theorem SimpContFract.of_isContFract :
    (SimpContFract.of v).IsContFract := fun _ _ nth_partDen_eq =>
  lt_of_lt_of_le zero_lt_one (of_one_le_get?_partDen nth_partDen_eq)


/-- Creates the continued fraction of a value. -/
def ContFract.of : ContFract K :=
  ⟨SimpContFract.of v, SimpContFract.of_isContFract v⟩


theorem fib_le_of_contsAux_b :
    n ≤ 1 ∨ ¬(of v).TerminatedAt (n - 2) → (fib n : K) ≤ ((of v).contsAux n).b :=
  Nat.strong_induction_on n
    (by
      /-
        K : Type u_1
        v : K
        n : Nat
        inst✝¹ : LinearOrderedField K
        inst✝ : FloorRing K
        ⊢ ∀ (n : Nat), (∀ (m : Nat), LT.lt m n → Or (LE.le m 1) (Not ((GenContFract.of …
      -/
      intro n IH hyp
      /-
        K : Type u_1
        v : K
        n✝ : Nat
        inst✝¹ : LinearOrderedField K
        inst✝ : FloorRing K
        n : Nat
        IH : ∀ (m : Nat), LT.lt m n → Or (LE.le m 1) (Not ((GenContFract.of v).Termina …
        hyp : Or (LE.le n 1) (Not ((GenContFract.of v).TerminatedAt (HSub.hSub n 2)))
        ⊢ LE.le (↑(Nat.fib n)) ((GenContFract.of v).contsAux n).b
      -/
      rcases n with (_ | _ | n)
        /-
          case zero
          K : Type u_1
          v : K
          n : Nat
          inst✝¹ : LinearOrderedField K
          inst✝ : FloorRing K
          IH : ∀ (m : Nat), LT.lt m 0 → Or (LE.le m 1) (Not ((GenContFract.of v).Termina …
          hyp : Or (LE.le 0 1) (Not ((GenContFract.of v).TerminatedAt (HSub.hSub 0 2)))
          ⊢ LE.le (↑(Nat.fib 0)) ((GenContFract.of v).contsAux 0).b
        -/
      · simp [fib_add_two, contsAux] -- case n = 0
        /-
          🎉 no goals
        -/
        /-
          case succ.zero
          K : Type u_1
          v : K
          n : Nat
          inst✝¹ : LinearOrderedField K
          inst✝ : FloorRing K
          IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd 0 1) → Or (LE.le m 1) (Not ((GenContFract …
          hyp : Or (LE.le (HAdd.hAdd 0 1) 1) (Not ((GenContFract.of v).TerminatedAt (HSu …
          ⊢ LE.le (↑(Nat.fib (HAdd.hAdd 0 1))) ((GenContFract.of v).contsAux (HAdd.hAdd  …
        -/
      · simp [fib_add_two, contsAux] -- case n = 1
        /-
          🎉 no goals
        -/
        /-
          case succ.succ
          K : Type u_1
          v : K
          n✝ : Nat
          inst✝¹ : LinearOrderedField K
          inst✝ : FloorRing K
          n : Nat
          IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd n 1) 1) → Or (LE.le m 1) (Not  …
          hyp : Or (LE.le (HAdd.hAdd (HAdd.hAdd n 1) 1) 1) (Not ((GenContFract.of v).Ter …
          ⊢ LE.le (↑(Nat.fib (HAdd.hAdd (HAdd.hAdd n 1) 1))) ((GenContFract.of v).contsA …
        -/
      · let g := of v -- case 2 ≤ n
        /-
          case succ.succ
          K : Type u_1
          v : K
          n✝ : Nat
          inst✝¹ : LinearOrderedField K
          inst✝ : FloorRing K
          n : Nat
          IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd n 1) 1) → Or (LE.le m 1) (Not  …
          hyp : Or (LE.le (HAdd.hAdd (HAdd.hAdd n 1) 1) 1) (Not ((GenContFract.of v).Ter …
          g : GenContFract K := GenContFract.of v
          ⊢ LE.le (↑(Nat.fib (HAdd.hAdd (HAdd.hAdd n 1) 1))) ((GenContFract.of v).contsA …
        -/
        have : ¬n + 2 ≤ 1 := by omega
        /-
          case succ.succ
          K : Type u_1
          v : K
          n✝ : Nat
          inst✝¹ : LinearOrderedField K
          inst✝ : FloorRing K
          n : Nat
          IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd n 1) 1) → Or (LE.le m 1) (Not  …
          hyp : Or (LE.le (HAdd.hAdd (HAdd.hAdd n 1) 1) 1) (Not ((GenContFract.of v).Ter …
          g : GenContFract K := GenContFract.of v
          this : Not (LE.le (HAdd.hAdd n 2) 1)
          ⊢ LE.le (↑(Nat.fib (HAdd.hAdd (HAdd.hAdd n 1) 1))) ((GenContFract.of v).contsA …
        -/
        have not_terminatedAt_n : ¬g.TerminatedAt n := Or.resolve_left hyp this
        obtain ⟨gp, s_ppred_nth_eq⟩ : ∃ gp, g.s.get? n = some gp :=
          Option.ne_none_iff_exists'.mp not_terminatedAt_n
        /-
          case succ.succ.intro
          K : Type u_1
          v : K
          n✝ : Nat
          inst✝¹ : LinearOrderedField K
          inst✝ : FloorRing K
          n : Nat
          IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd n 1) 1) → Or (LE.le m 1) (Not  …
          hyp : Or (LE.le (HAdd.hAdd (HAdd.hAdd n 1) 1) 1) (Not ((GenContFract.of v).Ter …
          g : GenContFract K := GenContFract.of v
          this : Not (LE.le (HAdd.hAdd n 2) 1)
          not_terminatedAt_n : Not (g.TerminatedAt n)
          gp : GenContFract.Pair K
          s_ppred_nth_eq : Eq (g.s.get? n) (Option.some gp)
          ⊢ LE.le (↑(Nat.fib (HAdd.hAdd (HAdd.hAdd n 1) 1))) ((GenContFract.of v).contsA …
        -/
        set pconts := g.contsAux (n + 1) with pconts_eq
        /-
          case succ.succ.intro
          K : Type u_1
          v : K
          n✝ : Nat
          inst✝¹ : LinearOrderedField K
          inst✝ : FloorRing K
          n : Nat
          IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd n 1) 1) → Or (LE.le m 1) (Not  …
          hyp : Or (LE.le (HAdd.hAdd (HAdd.hAdd n 1) 1) 1) (Not ((GenContFract.of v).Ter …
          g : GenContFract K := GenContFract.of v
          this : Not (LE.le (HAdd.hAdd n 2) 1)
          not_terminatedAt_n : Not (g.TerminatedAt n)
          gp : GenContFract.Pair K
          s_ppred_nth_eq : Eq (g.s.get? n) (Option.some gp)
          pconts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
          pconts_eq : Eq pconts (g.contsAux (HAdd.hAdd n 1))
          ⊢ LE.le (↑(Nat.fib (HAdd.hAdd (HAdd.hAdd n 1) 1))) ((GenContFract.of v).contsA …
        -/
        set ppconts := g.contsAux n with ppconts_eq
        -- use the recurrence of `contsAux`
        /-
          case succ.succ.intro
          K : Type u_1
          v : K
          n✝ : Nat
          inst✝¹ : LinearOrderedField K
          inst✝ : FloorRing K
          n : Nat
          IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd n 1) 1) → Or (LE.le m 1) (Not  …
          hyp : Or (LE.le (HAdd.hAdd (HAdd.hAdd n 1) 1) 1) (Not ((GenContFract.of v).Ter …
          g : GenContFract K := GenContFract.of v
          this : Not (LE.le (HAdd.hAdd n 2) 1)
          not_terminatedAt_n : Not (g.TerminatedAt n)
          gp : GenContFract.Pair K
          s_ppred_nth_eq : Eq (g.s.get? n) (Option.some gp)
          pconts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
          pconts_eq : Eq pconts (g.contsAux (HAdd.hAdd n 1))
          ppconts : GenContFract.Pair K := g.contsAux n
          ppconts_eq : Eq ppconts (g.contsAux n)
          ⊢ LE.le (↑(Nat.fib (HAdd.hAdd (HAdd.hAdd n 1) 1))) ((GenContFract.of v).contsA …
        -/
        simp only [Nat.succ_eq_add_one, Nat.add_assoc, Nat.reduceAdd]
        suffices (fib n : K) + fib (n + 1) ≤ gp.a * ppconts.b + gp.b * pconts.b by
          simpa [g, fib_add_two, add_comm, contsAux_recurrence s_ppred_nth_eq ppconts_eq pconts_eq]
        -- make use of the fact that `gp.a = 1`
        suffices (fib n : K) + fib (n + 1) ≤ ppconts.b + gp.b * pconts.b by
          simpa [of_partNum_eq_one <| partNum_eq_s_a s_ppred_nth_eq]
        have not_terminatedAt_pred_n : ¬g.TerminatedAt (n - 1) :=
          mt (terminated_stable <| Nat.sub_le n 1) not_terminatedAt_n
        have not_terminatedAt_ppred_n : ¬TerminatedAt g (n - 2) :=
          mt (terminated_stable (n - 1).pred_le) not_terminatedAt_pred_n
        -- use the IH to get the inequalities for `pconts` and `ppconts`
        have ppred_nth_fib_le_ppconts_B : (fib n : K) ≤ ppconts.b :=
          IH n (lt_trans (Nat.lt.base n) <| Nat.lt.base <| n + 1) (Or.inr not_terminatedAt_ppred_n)
        suffices (fib (n + 1) : K) ≤ gp.b * pconts.b by
          solve_by_elim [_root_.add_le_add ppred_nth_fib_le_ppconts_B]
        -- finally use the fact that `1 ≤ gp.b` to solve the goal
        /-
          case succ.succ.intro
          K : Type u_1
          v : K
          n✝ : Nat
          inst✝¹ : LinearOrderedField K
          inst✝ : FloorRing K
          n : Nat
          IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd n 1) 1) → Or (LE.le m 1) (Not  …
          hyp : Or (LE.le (HAdd.hAdd (HAdd.hAdd n 1) 1) 1) (Not ((GenContFract.of v).Ter …
          g : GenContFract K := GenContFract.of v
          this : Not (LE.le (HAdd.hAdd n 2) 1)
          not_terminatedAt_n : Not (g.TerminatedAt n)
          gp : GenContFract.Pair K
          s_ppred_nth_eq : Eq (g.s.get? n) (Option.some gp)
          pconts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
          pconts_eq : Eq pconts (g.contsAux (HAdd.hAdd n 1))
          ppconts : GenContFract.Pair K := g.contsAux n
          ppconts_eq : Eq ppconts (g.contsAux n)
          not_terminatedAt_pred_n : Not (g.TerminatedAt (HSub.hSub n 1))
          not_terminatedAt_ppred_n : Not (g.TerminatedAt (HSub.hSub n 2))
          ppred_nth_fib_le_ppconts_B : LE.le (↑(Nat.fib n)) ppconts.b
          ⊢ LE.le (↑(Nat.fib (HAdd.hAdd n 1))) (HMul.hMul gp.b pconts.b)
        -/
        suffices 1 * (fib (n + 1) : K) ≤ gp.b * pconts.b by rwa [one_mul] at this
        have one_le_gp_b : (1 : K) ≤ gp.b :=
          of_one_le_get?_partDen (partDen_eq_s_b s_ppred_nth_eq)
        /-
          case succ.succ.intro
          K : Type u_1
          v : K
          n✝ : Nat
          inst✝¹ : LinearOrderedField K
          inst✝ : FloorRing K
          n : Nat
          IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd n 1) 1) → Or (LE.le m 1) (Not  …
          hyp : Or (LE.le (HAdd.hAdd (HAdd.hAdd n 1) 1) 1) (Not ((GenContFract.of v).Ter …
          g : GenContFract K := GenContFract.of v
          this : Not (LE.le (HAdd.hAdd n 2) 1)
          not_terminatedAt_n : Not (g.TerminatedAt n)
          gp : GenContFract.Pair K
          s_ppred_nth_eq : Eq (g.s.get? n) (Option.some gp)
          pconts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
          pconts_eq : Eq pconts (g.contsAux (HAdd.hAdd n 1))
          ppconts : GenContFract.Pair K := g.contsAux n
          ppconts_eq : Eq ppconts (g.contsAux n)
          not_terminatedAt_pred_n : Not (g.TerminatedAt (HSub.hSub n 1))
          not_terminatedAt_ppred_n : Not (g.TerminatedAt (HSub.hSub n 2))
          ppred_nth_fib_le_ppconts_B : LE.le (↑(Nat.fib n)) ppconts.b
          one_le_gp_b : LE.le 1 gp.b
          ⊢ LE.le (HMul.hMul 1 ↑(Nat.fib (HAdd.hAdd n 1))) (HMul.hMul gp.b pconts.b)
        -/
        have : (0 : K) ≤ fib (n + 1) := mod_cast (fib (n + 1)).zero_le
        /-
          case succ.succ.intro
          K : Type u_1
          v : K
          n✝ : Nat
          inst✝¹ : LinearOrderedField K
          inst✝ : FloorRing K
          n : Nat
          IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd n 1) 1) → Or (LE.le m 1) (Not  …
          hyp : Or (LE.le (HAdd.hAdd (HAdd.hAdd n 1) 1) 1) (Not ((GenContFract.of v).Ter …
          g : GenContFract K := GenContFract.of v
          this✝ : Not (LE.le (HAdd.hAdd n 2) 1)
          not_terminatedAt_n : Not (g.TerminatedAt n)
          gp : GenContFract.Pair K
          s_ppred_nth_eq : Eq (g.s.get? n) (Option.some gp)
          pconts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
          pconts_eq : Eq pconts (g.contsAux (HAdd.hAdd n 1))
          ppconts : GenContFract.Pair K := g.contsAux n
          ppconts_eq : Eq ppconts (g.contsAux n)
          not_terminatedAt_pred_n : Not (g.TerminatedAt (HSub.hSub n 1))
          not_terminatedAt_ppred_n : Not (g.TerminatedAt (HSub.hSub n 2))
          ppred_nth_fib_le_ppconts_B : LE.le (↑(Nat.fib n)) ppconts.b
          one_le_gp_b : LE.le 1 gp.b
          this : LE.le 0 ↑(Nat.fib (HAdd.hAdd n 1))
          ⊢ LE.le (HMul.hMul 1 ↑(Nat.fib (HAdd.hAdd n 1))) (HMul.hMul gp.b pconts.b)
        -/
        have : (0 : K) ≤ gp.b := le_trans zero_le_one one_le_gp_b
        /-
          case succ.succ.intro
          K : Type u_1
          v : K
          n✝ : Nat
          inst✝¹ : LinearOrderedField K
          inst✝ : FloorRing K
          n : Nat
          IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd n 1) 1) → Or (LE.le m 1) (Not  …
          hyp : Or (LE.le (HAdd.hAdd (HAdd.hAdd n 1) 1) 1) (Not ((GenContFract.of v).Ter …
          g : GenContFract K := GenContFract.of v
          this✝¹ : Not (LE.le (HAdd.hAdd n 2) 1)
          not_terminatedAt_n : Not (g.TerminatedAt n)
          gp : GenContFract.Pair K
          s_ppred_nth_eq : Eq (g.s.get? n) (Option.some gp)
          pconts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
          pconts_eq : Eq pconts (g.contsAux (HAdd.hAdd n 1))
          ppconts : GenContFract.Pair K := g.contsAux n
          ppconts_eq : Eq ppconts (g.contsAux n)
          not_terminatedAt_pred_n : Not (g.TerminatedAt (HSub.hSub n 1))
          not_terminatedAt_ppred_n : Not (g.TerminatedAt (HSub.hSub n 2))
          ppred_nth_fib_le_ppconts_B : LE.le (↑(Nat.fib n)) ppconts.b
          one_le_gp_b : LE.le 1 gp.b
          this✝ : LE.le 0 ↑(Nat.fib (HAdd.hAdd n 1))
          this : LE.le 0 gp.b
          ⊢ LE.le (HMul.hMul 1 ↑(Nat.fib (HAdd.hAdd n 1))) (HMul.hMul gp.b pconts.b)
        -/
        mono
          /-
            case succ.succ.intro.h₂.a
            K : Type u_1
            v : K
            n✝ : Nat
            inst✝¹ : LinearOrderedField K
            inst✝ : FloorRing K
            n : Nat
            IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd n 1) 1) → Or (LE.le m 1) (Not  …
            hyp : Or (LE.le (HAdd.hAdd (HAdd.hAdd n 1) 1) 1) (Not ((GenContFract.of v).Ter …
            g : GenContFract K := GenContFract.of v
            this✝¹ : Not (LE.le (HAdd.hAdd n 2) 1)
            not_terminatedAt_n : Not (g.TerminatedAt n)
            gp : GenContFract.Pair K
            s_ppred_nth_eq : Eq (g.s.get? n) (Option.some gp)
            pconts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
            pconts_eq : Eq pconts (g.contsAux (HAdd.hAdd n 1))
            ppconts : GenContFract.Pair K := g.contsAux n
            ppconts_eq : Eq ppconts (g.contsAux n)
            not_terminatedAt_pred_n : Not (g.TerminatedAt (HSub.hSub n 1))
            not_terminatedAt_ppred_n : Not (g.TerminatedAt (HSub.hSub n 2))
            ppred_nth_fib_le_ppconts_B : LE.le (↑(Nat.fib n)) ppconts.b
            one_le_gp_b : LE.le 1 gp.b
            this✝ : LE.le 0 ↑(Nat.fib (HAdd.hAdd n 1))
            this : LE.le 0 gp.b
            ⊢ LT.lt (HAdd.hAdd n 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
          -/
        · norm_num
          /-
            🎉 no goals
          -/
          /-
            case succ.succ.intro.h₂.a
            K : Type u_1
            v : K
            n✝ : Nat
            inst✝¹ : LinearOrderedField K
            inst✝ : FloorRing K
            n : Nat
            IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd n 1) 1) → Or (LE.le m 1) (Not  …
            hyp : Or (LE.le (HAdd.hAdd (HAdd.hAdd n 1) 1) 1) (Not ((GenContFract.of v).Ter …
            g : GenContFract K := GenContFract.of v
            this✝¹ : Not (LE.le (HAdd.hAdd n 2) 1)
            not_terminatedAt_n : Not (g.TerminatedAt n)
            gp : GenContFract.Pair K
            s_ppred_nth_eq : Eq (g.s.get? n) (Option.some gp)
            pconts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
            pconts_eq : Eq pconts (g.contsAux (HAdd.hAdd n 1))
            ppconts : GenContFract.Pair K := g.contsAux n
            ppconts_eq : Eq ppconts (g.contsAux n)
            not_terminatedAt_pred_n : Not (g.TerminatedAt (HSub.hSub n 1))
            not_terminatedAt_ppred_n : Not (g.TerminatedAt (HSub.hSub n 2))
            ppred_nth_fib_le_ppconts_B : LE.le (↑(Nat.fib n)) ppconts.b
            one_le_gp_b : LE.le 1 gp.b
            this✝ : LE.le 0 ↑(Nat.fib (HAdd.hAdd n 1))
            this : LE.le 0 gp.b
            ⊢ Or (LE.le (HAdd.hAdd n 1) 1) (Not ((GenContFract.of v).TerminatedAt (HSub.hS …
          -/
        · tauto)
          /-
            🎉 no goals
          -/


/-- Shows that the `n`th denominator is greater than or equal to the `n + 1`th fibonacci number,
that is `Nat.fib (n + 1) ≤ Bₙ`. -/
theorem succ_nth_fib_le_of_nth_den (hyp : n = 0 ∨ ¬(of v).TerminatedAt (n - 1)) :
    (fib (n + 1) : K) ≤ (of v).dens n := by
  /-
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    hyp : Or (Eq n 0) (Not ((GenContFract.of v).TerminatedAt (HSub.hSub n 1)))
    ⊢ LE.le (↑(Nat.fib (HAdd.hAdd n 1))) ((GenContFract.of v).dens n)
  -/
  rw [den_eq_conts_b, nth_cont_eq_succ_nth_contAux]
  have : n + 1 ≤ 1 ∨ ¬(of v).TerminatedAt (n - 1) := by
    cases n with
    | zero => exact Or.inl <| le_refl 1
    | succ n => exact Or.inr (Or.resolve_left hyp n.succ_ne_zero)
  /-
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    hyp : Or (Eq n 0) (Not ((GenContFract.of v).TerminatedAt (HSub.hSub n 1)))
    this : Or (LE.le (HAdd.hAdd n 1) 1) (Not ((GenContFract.of v).TerminatedAt (HS …
    ⊢ LE.le (↑(Nat.fib (HAdd.hAdd n 1))) ((GenContFract.of v).contsAux (HAdd.hAdd  …
  -/
  exact fib_le_of_contsAux_b this
  /-
    🎉 no goals
  -/


theorem zero_le_of_contsAux_b : 0 ≤ ((of v).contsAux n).b := by
  /-
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    ⊢ LE.le 0 ((GenContFract.of v).contsAux n).b
  -/
  let g := of v
  induction n with
  | zero => rfl
  | succ n IH =>
    rcases Decidable.em <| g.TerminatedAt (n - 1) with terminated | not_terminated
    · -- terminating case
      rcases n with - | n
      · simp [zero_le_one]
      · have : g.contsAux (n + 2) = g.contsAux (n + 1) :=
          contsAux_stable_step_of_terminated terminated
        simp only [g, this, IH]
    · -- non-terminating case
      calc
        (0 : K) ≤ fib (n + 1) := mod_cast (n + 1).fib.zero_le
        _ ≤ ((of v).contsAux (n + 1)).b := fib_le_of_contsAux_b (Or.inr not_terminated)


/-- Shows that all denominators are nonnegative. -/
theorem zero_le_of_den : 0 ≤ (of v).dens n := by
  /-
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    ⊢ LE.le 0 ((GenContFract.of v).dens n)
  -/
  rw [den_eq_conts_b, nth_cont_eq_succ_nth_contAux]; exact zero_le_of_contsAux_b
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem le_of_succ_succ_get?_contsAux_b {b : K}
    (nth_partDen_eq : (of v).partDens.get? n = some b) :
    b * ((of v).contsAux <| n + 1).b ≤ ((of v).contsAux <| n + 2).b := by
  obtain ⟨gp_n, nth_s_eq, rfl⟩ : ∃ gp_n, (of v).s.get? n = some gp_n ∧ gp_n.b = b :=
    exists_s_b_of_partDen nth_partDen_eq
  simp [of_partNum_eq_one (partNum_eq_s_a nth_s_eq), zero_le_of_contsAux_b,
    GenContFract.contsAux_recurrence nth_s_eq rfl rfl]


/-- Shows that `bₙ * Bₙ ≤ Bₙ₊₁`, where `bₙ` is the `n`th partial denominator and `Bₙ₊₁` and `Bₙ` are
the `n + 1`th and `n`th denominator of the continued fraction. -/
theorem le_of_succ_get?_den {b : K}
    (nth_partDenom_eq : (of v).partDens.get? n = some b) :
    b * (of v).dens n ≤ (of v).dens (n + 1) := by
  /-
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    b : K
    nth_partDenom_eq : Eq ((GenContFract.of v).partDens.get? n) (Option.some b)
    ⊢ LE.le (HMul.hMul b ((GenContFract.of v).dens n)) ((GenContFract.of v).dens ( …
  -/
  rw [den_eq_conts_b, nth_cont_eq_succ_nth_contAux]
  /-
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    b : K
    nth_partDenom_eq : Eq ((GenContFract.of v).partDens.get? n) (Option.some b)
    ⊢ LE.le (HMul.hMul b ((GenContFract.of v).contsAux (HAdd.hAdd n 1)).b) ((GenCo …
  -/
  exact le_of_succ_succ_get?_contsAux_b nth_partDenom_eq
  /-
    🎉 no goals
  -/


/-- Shows that the sequence of denominators is monotone, that is `Bₙ ≤ Bₙ₊₁`. -/
theorem of_den_mono : (of v).dens n ≤ (of v).dens (n + 1) := by
  /-
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    ⊢ LE.le ((GenContFract.of v).dens n) ((GenContFract.of v).dens (HAdd.hAdd n 1))
  -/
  let g := of v
  /-
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    g : GenContFract K := GenContFract.of v
    ⊢ LE.le ((GenContFract.of v).dens n) ((GenContFract.of v).dens (HAdd.hAdd n 1))
  -/
  rcases Decidable.em <| g.partDens.TerminatedAt n with terminated | not_terminated
    /-
      case inl
      K : Type u_1
      v : K
      n : Nat
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      g : GenContFract K := GenContFract.of v
      terminated : g.partDens.TerminatedAt n
      ⊢ LE.le ((GenContFract.of v).dens n) ((GenContFract.of v).dens (HAdd.hAdd n 1))
    -/
  · have : g.partDens.get? n = none := by rwa [Stream'.Seq.TerminatedAt] at terminated
    have : g.TerminatedAt n :=
      terminatedAt_iff_partDen_none.2 (by rwa [Stream'.Seq.TerminatedAt] at terminated)
    have : g.dens (n + 1) = g.dens n :=
      dens_stable_of_terminated n.le_succ this
    /-
      case inl
      K : Type u_1
      v : K
      n : Nat
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      g : GenContFract K := GenContFract.of v
      terminated : g.partDens.TerminatedAt n
      this✝¹ : Eq (g.partDens.get? n) Option.none
      this✝ : g.TerminatedAt n
      this : Eq (g.dens (HAdd.hAdd n 1)) (g.dens n)
      ⊢ LE.le ((GenContFract.of v).dens n) ((GenContFract.of v).dens (HAdd.hAdd n 1))
    -/
    rw [this]
    /-
      🎉 no goals
    -/
  · obtain ⟨b, nth_partDen_eq⟩ : ∃ b, g.partDens.get? n = some b :=
      Option.ne_none_iff_exists'.mp not_terminated
    /-
      case inr.intro
      K : Type u_1
      v : K
      n : Nat
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      g : GenContFract K := GenContFract.of v
      not_terminated : Not (g.partDens.TerminatedAt n)
      b : K
      nth_partDen_eq : Eq (g.partDens.get? n) (Option.some b)
      ⊢ LE.le ((GenContFract.of v).dens n) ((GenContFract.of v).dens (HAdd.hAdd n 1))
    -/
    have : 1 ≤ b := of_one_le_get?_partDen nth_partDen_eq
    calc
      g.dens n ≤ b * g.dens n := by
        simpa using mul_le_mul_of_nonneg_right this zero_le_of_den
      _ ≤ g.dens (n + 1) := le_of_succ_get?_den nth_partDen_eq


/-- This lemma follows from the finite correctness proof, the determinant equality, and
by simplifying the difference. -/
theorem sub_convs_eq {ifp : IntFractPair K}
    (stream_nth_eq : IntFractPair.stream v n = some ifp) :
    let g := of v
    let B := (g.contsAux (n + 1)).b
    let pB := (g.contsAux n).b
    v - g.convs n = if ifp.fr = 0 then 0 else (-1) ^ n / (B * (ifp.fr⁻¹ * B + pB)) := by
  -- set up some shorthand notation
  /-
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    ifp : GenContFract.IntFractPair K
    stream_nth_eq : Eq (GenContFract.IntFractPair.stream v n) (Option.some ifp)
    ⊢ let g := GenContFract.of v;
      let B := (g.contsAux (HAdd.hAdd n 1)).b;
      let pB := (g.contsAux n).b;
      Eq (HSub.hSub v (g.convs n)) (ite (Eq ifp.fr 0) 0 (HDiv.hDiv (HPow.hPow (-1) …
  -/
  let g := of v
  /-
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    ifp : GenContFract.IntFractPair K
    stream_nth_eq : Eq (GenContFract.IntFractPair.stream v n) (Option.some ifp)
    g : GenContFract K := GenContFract.of v
    ⊢ let g := GenContFract.of v;
      let B := (g.contsAux (HAdd.hAdd n 1)).b;
      let pB := (g.contsAux n).b;
      Eq (HSub.hSub v (g.convs n)) (ite (Eq ifp.fr 0) 0 (HDiv.hDiv (HPow.hPow (-1) …
  -/
  let conts := g.contsAux (n + 1)
  /-
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    ifp : GenContFract.IntFractPair K
    stream_nth_eq : Eq (GenContFract.IntFractPair.stream v n) (Option.some ifp)
    g : GenContFract K := GenContFract.of v
    conts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
    ⊢ let g := GenContFract.of v;
      let B := (g.contsAux (HAdd.hAdd n 1)).b;
      let pB := (g.contsAux n).b;
      Eq (HSub.hSub v (g.convs n)) (ite (Eq ifp.fr 0) 0 (HDiv.hDiv (HPow.hPow (-1) …
  -/
  let pred_conts := g.contsAux n
  have g_finite_correctness :
    v = GenContFract.compExactValue pred_conts conts ifp.fr :=
    compExactValue_correctness_of_stream_eq_some stream_nth_eq
  /-
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    ifp : GenContFract.IntFractPair K
    stream_nth_eq : Eq (GenContFract.IntFractPair.stream v n) (Option.some ifp)
    g : GenContFract K := GenContFract.of v
    conts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
    pred_conts : GenContFract.Pair K := g.contsAux n
    g_finite_correctness : Eq v (GenContFract.compExactValue pred_conts conts ifp. …
    ⊢ let g := GenContFract.of v;
      let B := (g.contsAux (HAdd.hAdd n 1)).b;
      let pB := (g.contsAux n).b;
      Eq (HSub.hSub v (g.convs n)) (ite (Eq ifp.fr 0) 0 (HDiv.hDiv (HPow.hPow (-1) …
  -/
  obtain (ifp_fr_eq_zero | ifp_fr_ne_zero) := eq_or_ne ifp.fr 0
    /-
      case inl
      K : Type u_1
      v : K
      n : Nat
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      ifp : GenContFract.IntFractPair K
      stream_nth_eq : Eq (GenContFract.IntFractPair.stream v n) (Option.some ifp)
      g : GenContFract K := GenContFract.of v
      conts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
      pred_conts : GenContFract.Pair K := g.contsAux n
      g_finite_correctness : Eq v (GenContFract.compExactValue pred_conts conts ifp. …
      ifp_fr_eq_zero : Eq ifp.fr 0
      ⊢ let g := GenContFract.of v;
        let B := (g.contsAux (HAdd.hAdd n 1)).b;
        let pB := (g.contsAux n).b;
        Eq (HSub.hSub v (g.convs n)) (ite (Eq ifp.fr 0) 0 (HDiv.hDiv (HPow.hPow (-1) …
    -/
  · suffices v - g.convs n = 0 by simpa [ifp_fr_eq_zero]
    replace g_finite_correctness : v = g.convs n := by
      simpa [GenContFract.compExactValue, ifp_fr_eq_zero] using g_finite_correctness
    /-
      case inl
      K : Type u_1
      v : K
      n : Nat
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      ifp : GenContFract.IntFractPair K
      stream_nth_eq : Eq (GenContFract.IntFractPair.stream v n) (Option.some ifp)
      g : GenContFract K := GenContFract.of v
      conts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
      pred_conts : GenContFract.Pair K := g.contsAux n
      ifp_fr_eq_zero : Eq ifp.fr 0
      g_finite_correctness : Eq v (g.convs n)
      ⊢ Eq (HSub.hSub v (g.convs n)) 0
    -/
    exact sub_eq_zero.2 g_finite_correctness
    /-
      🎉 no goals
    -/
  · -- more shorthand notation
    /-
      case inr
      K : Type u_1
      v : K
      n : Nat
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      ifp : GenContFract.IntFractPair K
      stream_nth_eq : Eq (GenContFract.IntFractPair.stream v n) (Option.some ifp)
      g : GenContFract K := GenContFract.of v
      conts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
      pred_conts : GenContFract.Pair K := g.contsAux n
      g_finite_correctness : Eq v (GenContFract.compExactValue pred_conts conts ifp. …
      ifp_fr_ne_zero : Ne ifp.fr 0
      ⊢ let g := GenContFract.of v;
        let B := (g.contsAux (HAdd.hAdd n 1)).b;
        let pB := (g.contsAux n).b;
        Eq (HSub.hSub v (g.convs n)) (ite (Eq ifp.fr 0) 0 (HDiv.hDiv (HPow.hPow (-1) …
    -/
    let A := conts.a
    /-
      case inr
      K : Type u_1
      v : K
      n : Nat
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      ifp : GenContFract.IntFractPair K
      stream_nth_eq : Eq (GenContFract.IntFractPair.stream v n) (Option.some ifp)
      g : GenContFract K := GenContFract.of v
      conts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
      pred_conts : GenContFract.Pair K := g.contsAux n
      g_finite_correctness : Eq v (GenContFract.compExactValue pred_conts conts ifp. …
      ifp_fr_ne_zero : Ne ifp.fr 0
      A : K := conts.a
      ⊢ let g := GenContFract.of v;
        let B := (g.contsAux (HAdd.hAdd n 1)).b;
        let pB := (g.contsAux n).b;
        Eq (HSub.hSub v (g.convs n)) (ite (Eq ifp.fr 0) 0 (HDiv.hDiv (HPow.hPow (-1) …
    -/
    let B := conts.b
    /-
      case inr
      K : Type u_1
      v : K
      n : Nat
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      ifp : GenContFract.IntFractPair K
      stream_nth_eq : Eq (GenContFract.IntFractPair.stream v n) (Option.some ifp)
      g : GenContFract K := GenContFract.of v
      conts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
      pred_conts : GenContFract.Pair K := g.contsAux n
      g_finite_correctness : Eq v (GenContFract.compExactValue pred_conts conts ifp. …
      ifp_fr_ne_zero : Ne ifp.fr 0
      A : K := conts.a
      B : K := conts.b
      ⊢ let g := GenContFract.of v;
        let B := (g.contsAux (HAdd.hAdd n 1)).b;
        let pB := (g.contsAux n).b;
        Eq (HSub.hSub v (g.convs n)) (ite (Eq ifp.fr 0) 0 (HDiv.hDiv (HPow.hPow (-1) …
    -/
    let pA := pred_conts.a
    /-
      case inr
      K : Type u_1
      v : K
      n : Nat
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      ifp : GenContFract.IntFractPair K
      stream_nth_eq : Eq (GenContFract.IntFractPair.stream v n) (Option.some ifp)
      g : GenContFract K := GenContFract.of v
      conts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
      pred_conts : GenContFract.Pair K := g.contsAux n
      g_finite_correctness : Eq v (GenContFract.compExactValue pred_conts conts ifp. …
      ifp_fr_ne_zero : Ne ifp.fr 0
      A : K := conts.a
      B : K := conts.b
      pA : K := pred_conts.a
      ⊢ let g := GenContFract.of v;
        let B := (g.contsAux (HAdd.hAdd n 1)).b;
        let pB := (g.contsAux n).b;
        Eq (HSub.hSub v (g.convs n)) (ite (Eq ifp.fr 0) 0 (HDiv.hDiv (HPow.hPow (-1) …
    -/
    let pB := pred_conts.b
    -- first, let's simplify the goal as `ifp.fr ≠ 0`
    /-
      case inr
      K : Type u_1
      v : K
      n : Nat
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      ifp : GenContFract.IntFractPair K
      stream_nth_eq : Eq (GenContFract.IntFractPair.stream v n) (Option.some ifp)
      g : GenContFract K := GenContFract.of v
      conts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
      pred_conts : GenContFract.Pair K := g.contsAux n
      g_finite_correctness : Eq v (GenContFract.compExactValue pred_conts conts ifp. …
      ifp_fr_ne_zero : Ne ifp.fr 0
      A : K := conts.a
      B : K := conts.b
      pA : K := pred_conts.a
      pB : K := pred_conts.b
      ⊢ let g := GenContFract.of v;
        let B := (g.contsAux (HAdd.hAdd n 1)).b;
        let pB := (g.contsAux n).b;
        Eq (HSub.hSub v (g.convs n)) (ite (Eq ifp.fr 0) 0 (HDiv.hDiv (HPow.hPow (-1) …
    -/
    suffices v - A / B = (-1) ^ n / (B * (ifp.fr⁻¹ * B + pB)) by simpa [ifp_fr_ne_zero]
    -- now we can unfold `g.compExactValue` to derive the following equality for `v`
    replace g_finite_correctness : v = (pA + ifp.fr⁻¹ * A) / (pB + ifp.fr⁻¹ * B) := by
      simpa [GenContFract.compExactValue, ifp_fr_ne_zero, nextConts, nextNum, nextDen, add_comm]
        using g_finite_correctness
    -- let's rewrite this equality for `v` in our goal
    suffices
      (pA + ifp.fr⁻¹ * A) / (pB + ifp.fr⁻¹ * B) - A / B = (-1) ^ n / (B * (ifp.fr⁻¹ * B + pB)) by
      rwa [g_finite_correctness]
    -- To continue, we need use the determinant equality. So let's derive the needed hypothesis.
    have n_eq_zero_or_not_terminatedAt_pred_n : n = 0 ∨ ¬g.TerminatedAt (n - 1) := by
      rcases n with - | n'
      · simp
      · have : IntFractPair.stream v (n' + 1) ≠ none := by simp [stream_nth_eq]
        have : ¬g.TerminatedAt n' :=
          (not_congr of_terminatedAt_n_iff_succ_nth_intFractPair_stream_eq_none).2 this
        exact Or.inr this
    have determinant_eq : pA * B - pB * A = (-1) ^ n :=
      (SimpContFract.of v).determinant_aux n_eq_zero_or_not_terminatedAt_pred_n
    -- now all we got to do is to rewrite this equality in our goal and re-arrange terms;
    -- however, for this, we first have to derive quite a few tedious inequalities.
    have pB_ineq : (fib n : K) ≤ pB :=
      haveI : n ≤ 1 ∨ ¬g.TerminatedAt (n - 2) := by
        rcases n_eq_zero_or_not_terminatedAt_pred_n with n_eq_zero | not_terminatedAt_pred_n
        · simp [n_eq_zero]
        · exact Or.inr <| mt (terminated_stable (n - 1).pred_le) not_terminatedAt_pred_n
      fib_le_of_contsAux_b this
    have B_ineq : (fib (n + 1) : K) ≤ B :=
      haveI : n + 1 ≤ 1 ∨ ¬g.TerminatedAt (n + 1 - 2) := by
        rcases n_eq_zero_or_not_terminatedAt_pred_n with n_eq_zero | not_terminatedAt_pred_n
        · simp [n_eq_zero, le_refl]
        · exact Or.inr not_terminatedAt_pred_n
      fib_le_of_contsAux_b this
    /-
      case inr
      K : Type u_1
      v : K
      n : Nat
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      ifp : GenContFract.IntFractPair K
      stream_nth_eq : Eq (GenContFract.IntFractPair.stream v n) (Option.some ifp)
      g : GenContFract K := GenContFract.of v
      conts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
      pred_conts : GenContFract.Pair K := g.contsAux n
      ifp_fr_ne_zero : Ne ifp.fr 0
      A : K := conts.a
      B : K := conts.b
      pA : K := pred_conts.a
      pB : K := pred_conts.b
      g_finite_correctness : Eq v (HDiv.hDiv (HAdd.hAdd pA (HMul.hMul (Inv.inv ifp.f …
      n_eq_zero_or_not_terminatedAt_pred_n : Or (Eq n 0) (Not (g.TerminatedAt (HSub. …
      determinant_eq : Eq (HSub.hSub (HMul.hMul pA B) (HMul.hMul pB A)) (HPow.hPow ( …
      pB_ineq : LE.le (↑(Nat.fib n)) pB
      B_ineq : LE.le (↑(Nat.fib (HAdd.hAdd n 1))) B
      ⊢ Eq (HSub.hSub (HDiv.hDiv (HAdd.hAdd pA (HMul.hMul (Inv.inv ifp.fr) A)) (HAdd …
    -/
    have zero_lt_B : 0 < B := B_ineq.trans_lt' <| cast_pos.2 <| fib_pos.2 n.succ_pos
    /-
      case inr
      K : Type u_1
      v : K
      n : Nat
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      ifp : GenContFract.IntFractPair K
      stream_nth_eq : Eq (GenContFract.IntFractPair.stream v n) (Option.some ifp)
      g : GenContFract K := GenContFract.of v
      conts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
      pred_conts : GenContFract.Pair K := g.contsAux n
      ifp_fr_ne_zero : Ne ifp.fr 0
      A : K := conts.a
      B : K := conts.b
      pA : K := pred_conts.a
      pB : K := pred_conts.b
      g_finite_correctness : Eq v (HDiv.hDiv (HAdd.hAdd pA (HMul.hMul (Inv.inv ifp.f …
      n_eq_zero_or_not_terminatedAt_pred_n : Or (Eq n 0) (Not (g.TerminatedAt (HSub. …
      determinant_eq : Eq (HSub.hSub (HMul.hMul pA B) (HMul.hMul pB A)) (HPow.hPow ( …
      pB_ineq : LE.le (↑(Nat.fib n)) pB
      B_ineq : LE.le (↑(Nat.fib (HAdd.hAdd n 1))) B
      zero_lt_B : LT.lt 0 B
      ⊢ Eq (HSub.hSub (HDiv.hDiv (HAdd.hAdd pA (HMul.hMul (Inv.inv ifp.fr) A)) (HAdd …
    -/
    have : 0 ≤ pB := (cast_nonneg _).trans pB_ineq
    have : 0 < ifp.fr :=
      ifp_fr_ne_zero.lt_of_le' <| IntFractPair.nth_stream_fr_nonneg stream_nth_eq
    /-
      case inr
      K : Type u_1
      v : K
      n : Nat
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      ifp : GenContFract.IntFractPair K
      stream_nth_eq : Eq (GenContFract.IntFractPair.stream v n) (Option.some ifp)
      g : GenContFract K := GenContFract.of v
      conts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
      pred_conts : GenContFract.Pair K := g.contsAux n
      ifp_fr_ne_zero : Ne ifp.fr 0
      A : K := conts.a
      B : K := conts.b
      pA : K := pred_conts.a
      pB : K := pred_conts.b
      g_finite_correctness : Eq v (HDiv.hDiv (HAdd.hAdd pA (HMul.hMul (Inv.inv ifp.f …
      n_eq_zero_or_not_terminatedAt_pred_n : Or (Eq n 0) (Not (g.TerminatedAt (HSub. …
      determinant_eq : Eq (HSub.hSub (HMul.hMul pA B) (HMul.hMul pB A)) (HPow.hPow ( …
      pB_ineq : LE.le (↑(Nat.fib n)) pB
      B_ineq : LE.le (↑(Nat.fib (HAdd.hAdd n 1))) B
      zero_lt_B : LT.lt 0 B
      this✝ : LE.le 0 pB
      this : LT.lt 0 ifp.fr
      ⊢ Eq (HSub.hSub (HDiv.hDiv (HAdd.hAdd pA (HMul.hMul (Inv.inv ifp.fr) A)) (HAdd …
    -/
    have : pB + ifp.fr⁻¹ * B ≠ 0 := by positivity
    -- finally, let's do the rewriting
    calc
      (pA + ifp.fr⁻¹ * A) / (pB + ifp.fr⁻¹ * B) - A / B =
          ((pA + ifp.fr⁻¹ * A) * B - (pB + ifp.fr⁻¹ * B) * A) / ((pB + ifp.fr⁻¹ * B) * B) := by
        rw [div_sub_div _ _ this zero_lt_B.ne']
      _ = (pA * B + ifp.fr⁻¹ * A * B - (pB * A + ifp.fr⁻¹ * B * A)) / _ := by repeat' rw [add_mul]
      _ = (pA * B - pB * A) / ((pB + ifp.fr⁻¹ * B) * B) := by ring
      _ = (-1) ^ n / ((pB + ifp.fr⁻¹ * B) * B) := by rw [determinant_eq]
      _ = (-1) ^ n / (B * (ifp.fr⁻¹ * B + pB)) := by ac_rfl


/-- Shows that `|v - Aₙ / Bₙ| ≤ 1 / (Bₙ * Bₙ₊₁)`. -/
theorem abs_sub_convs_le (not_terminatedAt_n : ¬(of v).TerminatedAt n) :
    |v - (of v).convs n| ≤ 1 / ((of v).dens n * ((of v).dens <| n + 1)) := by
  -- shorthand notation
  /-
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    not_terminatedAt_n : Not ((GenContFract.of v).TerminatedAt n)
    ⊢ LE.le (abs (HSub.hSub v ((GenContFract.of v).convs n))) (HDiv.hDiv 1 (HMul.h …
  -/
  let g := of v
  /-
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    not_terminatedAt_n : Not ((GenContFract.of v).TerminatedAt n)
    g : GenContFract K := GenContFract.of v
    ⊢ LE.le (abs (HSub.hSub v ((GenContFract.of v).convs n))) (HDiv.hDiv 1 (HMul.h …
  -/
  let nextConts := g.contsAux (n + 2)
  /-
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    not_terminatedAt_n : Not ((GenContFract.of v).TerminatedAt n)
    g : GenContFract K := GenContFract.of v
    nextConts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 2)
    ⊢ LE.le (abs (HSub.hSub v ((GenContFract.of v).convs n))) (HDiv.hDiv 1 (HMul.h …
  -/
  set conts := contsAux g (n + 1) with conts_eq
  /-
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    not_terminatedAt_n : Not ((GenContFract.of v).TerminatedAt n)
    g : GenContFract K := GenContFract.of v
    nextConts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 2)
    conts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
    conts_eq : Eq conts (g.contsAux (HAdd.hAdd n 1))
    ⊢ LE.le (abs (HSub.hSub v ((GenContFract.of v).convs n))) (HDiv.hDiv 1 (HMul.h …
  -/
  set pred_conts := contsAux g n with pred_conts_eq
  -- change the goal to something more readable
  /-
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    not_terminatedAt_n : Not ((GenContFract.of v).TerminatedAt n)
    g : GenContFract K := GenContFract.of v
    nextConts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 2)
    conts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
    conts_eq : Eq conts (g.contsAux (HAdd.hAdd n 1))
    pred_conts : GenContFract.Pair K := g.contsAux n
    pred_conts_eq : Eq pred_conts (g.contsAux n)
    ⊢ LE.le (abs (HSub.hSub v ((GenContFract.of v).convs n))) (HDiv.hDiv 1 (HMul.h …
  -/
  change |v - convs g n| ≤ 1 / (conts.b * nextConts.b)
  obtain ⟨gp, s_nth_eq⟩ : ∃ gp, g.s.get? n = some gp :=
    Option.ne_none_iff_exists'.1 not_terminatedAt_n
  /-
    case intro
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    not_terminatedAt_n : Not ((GenContFract.of v).TerminatedAt n)
    g : GenContFract K := GenContFract.of v
    nextConts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 2)
    conts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
    conts_eq : Eq conts (g.contsAux (HAdd.hAdd n 1))
    pred_conts : GenContFract.Pair K := g.contsAux n
    pred_conts_eq : Eq pred_conts (g.contsAux n)
    gp : GenContFract.Pair K
    s_nth_eq : Eq (g.s.get? n) (Option.some gp)
    ⊢ LE.le (abs (HSub.hSub v (g.convs n))) (HDiv.hDiv 1 (HMul.hMul conts.b nextCo …
  -/
  have gp_a_eq_one : gp.a = 1 := of_partNum_eq_one (partNum_eq_s_a s_nth_eq)
  -- unfold the recurrence relation for `nextConts.b`
  have nextConts_b_eq : nextConts.b = pred_conts.b + gp.b * conts.b := by
    simp [nextConts, contsAux_recurrence s_nth_eq pred_conts_eq conts_eq, gp_a_eq_one,
      pred_conts_eq.symm, conts_eq.symm, add_comm]
  /-
    case intro
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    not_terminatedAt_n : Not ((GenContFract.of v).TerminatedAt n)
    g : GenContFract K := GenContFract.of v
    nextConts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 2)
    conts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
    conts_eq : Eq conts (g.contsAux (HAdd.hAdd n 1))
    pred_conts : GenContFract.Pair K := g.contsAux n
    pred_conts_eq : Eq pred_conts (g.contsAux n)
    gp : GenContFract.Pair K
    s_nth_eq : Eq (g.s.get? n) (Option.some gp)
    gp_a_eq_one : Eq gp.a 1
    nextConts_b_eq : Eq nextConts.b (HAdd.hAdd pred_conts.b (HMul.hMul gp.b conts. …
    ⊢ LE.le (abs (HSub.hSub v (g.convs n))) (HDiv.hDiv 1 (HMul.hMul conts.b nextCo …
  -/
  let den := conts.b * (pred_conts.b + gp.b * conts.b)
  /-
    case intro
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    not_terminatedAt_n : Not ((GenContFract.of v).TerminatedAt n)
    g : GenContFract K := GenContFract.of v
    nextConts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 2)
    conts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
    conts_eq : Eq conts (g.contsAux (HAdd.hAdd n 1))
    pred_conts : GenContFract.Pair K := g.contsAux n
    pred_conts_eq : Eq pred_conts (g.contsAux n)
    gp : GenContFract.Pair K
    s_nth_eq : Eq (g.s.get? n) (Option.some gp)
    gp_a_eq_one : Eq gp.a 1
    nextConts_b_eq : Eq nextConts.b (HAdd.hAdd pred_conts.b (HMul.hMul gp.b conts. …
    den : K := HMul.hMul conts.b (HAdd.hAdd pred_conts.b (HMul.hMul gp.b conts.b))
    ⊢ LE.le (abs (HSub.hSub v (g.convs n))) (HDiv.hDiv 1 (HMul.hMul conts.b nextCo …
  -/
  suffices |v - g.convs n| ≤ 1 / den by rw [nextConts_b_eq]; congr 1
  obtain ⟨ifp_succ_n, succ_nth_stream_eq, ifp_succ_n_b_eq_gp_b⟩ :
      ∃ ifp_succ_n, IntFractPair.stream v (n + 1) = some ifp_succ_n ∧ (ifp_succ_n.b : K) = gp.b :=
    IntFractPair.exists_succ_get?_stream_of_gcf_of_get?_eq_some s_nth_eq
  obtain ⟨ifp_n, stream_nth_eq, stream_nth_fr_ne_zero, if_of_eq_ifp_succ_n⟩ :
    ∃ ifp_n, IntFractPair.stream v n = some ifp_n ∧ ifp_n.fr ≠ 0
      ∧ IntFractPair.of ifp_n.fr⁻¹ = ifp_succ_n :=
    IntFractPair.succ_nth_stream_eq_some_iff.1 succ_nth_stream_eq
  /-
    case intro.intro.intro.intro.intro.intro
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    not_terminatedAt_n : Not ((GenContFract.of v).TerminatedAt n)
    g : GenContFract K := GenContFract.of v
    nextConts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 2)
    conts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
    conts_eq : Eq conts (g.contsAux (HAdd.hAdd n 1))
    pred_conts : GenContFract.Pair K := g.contsAux n
    pred_conts_eq : Eq pred_conts (g.contsAux n)
    gp : GenContFract.Pair K
    s_nth_eq : Eq (g.s.get? n) (Option.some gp)
    gp_a_eq_one : Eq gp.a 1
    nextConts_b_eq : Eq nextConts.b (HAdd.hAdd pred_conts.b (HMul.hMul gp.b conts. …
    den : K := HMul.hMul conts.b (HAdd.hAdd pred_conts.b (HMul.hMul gp.b conts.b))
    ifp_succ_n : GenContFract.IntFractPair K
    succ_nth_stream_eq : Eq (GenContFract.IntFractPair.stream v (HAdd.hAdd n 1)) ( …
    ifp_succ_n_b_eq_gp_b : Eq (↑ifp_succ_n.b) gp.b
    ifp_n : GenContFract.IntFractPair K
    stream_nth_eq : Eq (GenContFract.IntFractPair.stream v n) (Option.some ifp_n)
    stream_nth_fr_ne_zero : Ne ifp_n.fr 0
    if_of_eq_ifp_succ_n : Eq (GenContFract.IntFractPair.of (Inv.inv ifp_n.fr)) ifp …
    ⊢ LE.le (abs (HSub.hSub v (g.convs n))) (HDiv.hDiv 1 den)
  -/
  let den' := conts.b * (pred_conts.b + ifp_n.fr⁻¹ * conts.b)
  -- now we can use `sub_convs_eq` to simplify our goal
  suffices |(-1) ^ n / den'| ≤ 1 / den by
    have : v - g.convs n = (-1) ^ n / den' := by
      -- apply `sub_convs_eq` and simplify the result
      have tmp := sub_convs_eq stream_nth_eq
      simp only [stream_nth_fr_ne_zero, conts_eq.symm, pred_conts_eq.symm, if_false] at tmp
      rw [tmp]
      ring
    rwa [this]
  -- derive some tedious inequalities that we need to rewrite our goal
  have nextConts_b_ineq : (fib (n + 2) : K) ≤ pred_conts.b + gp.b * conts.b := by
    have : (fib (n + 2) : K) ≤ nextConts.b :=
      fib_le_of_contsAux_b (Or.inr not_terminatedAt_n)
    rwa [nextConts_b_eq] at this
  have conts_b_ineq : (fib (n + 1) : K) ≤ conts.b :=
    haveI : ¬g.TerminatedAt (n - 1) := mt (terminated_stable n.pred_le) not_terminatedAt_n
    fib_le_of_contsAux_b <| Or.inr this
  have zero_lt_conts_b : 0 < conts.b :=
    conts_b_ineq.trans_lt' <| mod_cast fib_pos.2 n.succ_pos
  -- `den'` is positive, so we can remove `|⬝|` from our goal
  suffices 1 / den' ≤ 1 / den by
    have : |(-1) ^ n / den'| = 1 / den' := by
      suffices 1 / |den'| = 1 / den' by rwa [abs_div, abs_neg_one_pow n]
      have : 0 < den' := by
        have : 0 ≤ pred_conts.b :=
          haveI : (fib n : K) ≤ pred_conts.b :=
            haveI : ¬g.TerminatedAt (n - 2) :=
              mt (terminated_stable (n.sub_le 2)) not_terminatedAt_n
            fib_le_of_contsAux_b <| Or.inr this
          le_trans (mod_cast (fib n).zero_le) this
        have : 0 < ifp_n.fr⁻¹ :=
          haveI zero_le_ifp_n_fract : 0 ≤ ifp_n.fr :=
            IntFractPair.nth_stream_fr_nonneg stream_nth_eq
          inv_pos.2 (lt_of_le_of_ne zero_le_ifp_n_fract stream_nth_fr_ne_zero.symm)
        -- Porting note: replaced complicated positivity proof with tactic.
        positivity
      rw [abs_of_pos this]
    rwa [this]
  /-
    case intro.intro.intro.intro.intro.intro
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    not_terminatedAt_n : Not ((GenContFract.of v).TerminatedAt n)
    g : GenContFract K := GenContFract.of v
    nextConts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 2)
    conts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
    conts_eq : Eq conts (g.contsAux (HAdd.hAdd n 1))
    pred_conts : GenContFract.Pair K := g.contsAux n
    pred_conts_eq : Eq pred_conts (g.contsAux n)
    gp : GenContFract.Pair K
    s_nth_eq : Eq (g.s.get? n) (Option.some gp)
    gp_a_eq_one : Eq gp.a 1
    nextConts_b_eq : Eq nextConts.b (HAdd.hAdd pred_conts.b (HMul.hMul gp.b conts. …
    den : K := HMul.hMul conts.b (HAdd.hAdd pred_conts.b (HMul.hMul gp.b conts.b))
    ifp_succ_n : GenContFract.IntFractPair K
    succ_nth_stream_eq : Eq (GenContFract.IntFractPair.stream v (HAdd.hAdd n 1)) ( …
    ifp_succ_n_b_eq_gp_b : Eq (↑ifp_succ_n.b) gp.b
    ifp_n : GenContFract.IntFractPair K
    stream_nth_eq : Eq (GenContFract.IntFractPair.stream v n) (Option.some ifp_n)
    stream_nth_fr_ne_zero : Ne ifp_n.fr 0
    if_of_eq_ifp_succ_n : Eq (GenContFract.IntFractPair.of (Inv.inv ifp_n.fr)) ifp …
    den' : K := HMul.hMul conts.b (HAdd.hAdd pred_conts.b (HMul.hMul (Inv.inv ifp_ …
    nextConts_b_ineq : LE.le (↑(Nat.fib (HAdd.hAdd n 2))) (HAdd.hAdd pred_conts.b  …
    conts_b_ineq : LE.le (↑(Nat.fib (HAdd.hAdd n 1))) conts.b
    zero_lt_conts_b : LT.lt 0 conts.b
    ⊢ LE.le (HDiv.hDiv 1 den') (HDiv.hDiv 1 den)
  -/
  suffices 0 < den ∧ den ≤ den' from div_le_div_of_nonneg_left zero_le_one this.1 this.2
  /-
    case intro.intro.intro.intro.intro.intro
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    not_terminatedAt_n : Not ((GenContFract.of v).TerminatedAt n)
    g : GenContFract K := GenContFract.of v
    nextConts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 2)
    conts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
    conts_eq : Eq conts (g.contsAux (HAdd.hAdd n 1))
    pred_conts : GenContFract.Pair K := g.contsAux n
    pred_conts_eq : Eq pred_conts (g.contsAux n)
    gp : GenContFract.Pair K
    s_nth_eq : Eq (g.s.get? n) (Option.some gp)
    gp_a_eq_one : Eq gp.a 1
    nextConts_b_eq : Eq nextConts.b (HAdd.hAdd pred_conts.b (HMul.hMul gp.b conts. …
    den : K := HMul.hMul conts.b (HAdd.hAdd pred_conts.b (HMul.hMul gp.b conts.b))
    ifp_succ_n : GenContFract.IntFractPair K
    succ_nth_stream_eq : Eq (GenContFract.IntFractPair.stream v (HAdd.hAdd n 1)) ( …
    ifp_succ_n_b_eq_gp_b : Eq (↑ifp_succ_n.b) gp.b
    ifp_n : GenContFract.IntFractPair K
    stream_nth_eq : Eq (GenContFract.IntFractPair.stream v n) (Option.some ifp_n)
    stream_nth_fr_ne_zero : Ne ifp_n.fr 0
    if_of_eq_ifp_succ_n : Eq (GenContFract.IntFractPair.of (Inv.inv ifp_n.fr)) ifp …
    den' : K := HMul.hMul conts.b (HAdd.hAdd pred_conts.b (HMul.hMul (Inv.inv ifp_ …
    nextConts_b_ineq : LE.le (↑(Nat.fib (HAdd.hAdd n 2))) (HAdd.hAdd pred_conts.b  …
    conts_b_ineq : LE.le (↑(Nat.fib (HAdd.hAdd n 1))) conts.b
    zero_lt_conts_b : LT.lt 0 conts.b
    ⊢ And (LT.lt 0 den) (LE.le den den')
  -/
  constructor
  · have : 0 < pred_conts.b + gp.b * conts.b :=
      nextConts_b_ineq.trans_lt' <| mod_cast fib_pos.2 <| succ_pos _
    /-
      case intro.intro.intro.intro.intro.intro.left
      K : Type u_1
      v : K
      n : Nat
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      not_terminatedAt_n : Not ((GenContFract.of v).TerminatedAt n)
      g : GenContFract K := GenContFract.of v
      nextConts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 2)
      conts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
      conts_eq : Eq conts (g.contsAux (HAdd.hAdd n 1))
      pred_conts : GenContFract.Pair K := g.contsAux n
      pred_conts_eq : Eq pred_conts (g.contsAux n)
      gp : GenContFract.Pair K
      s_nth_eq : Eq (g.s.get? n) (Option.some gp)
      gp_a_eq_one : Eq gp.a 1
      nextConts_b_eq : Eq nextConts.b (HAdd.hAdd pred_conts.b (HMul.hMul gp.b conts. …
      den : K := HMul.hMul conts.b (HAdd.hAdd pred_conts.b (HMul.hMul gp.b conts.b))
      ifp_succ_n : GenContFract.IntFractPair K
      succ_nth_stream_eq : Eq (GenContFract.IntFractPair.stream v (HAdd.hAdd n 1)) ( …
      ifp_succ_n_b_eq_gp_b : Eq (↑ifp_succ_n.b) gp.b
      ifp_n : GenContFract.IntFractPair K
      stream_nth_eq : Eq (GenContFract.IntFractPair.stream v n) (Option.some ifp_n)
      stream_nth_fr_ne_zero : Ne ifp_n.fr 0
      if_of_eq_ifp_succ_n : Eq (GenContFract.IntFractPair.of (Inv.inv ifp_n.fr)) ifp …
      den' : K := HMul.hMul conts.b (HAdd.hAdd pred_conts.b (HMul.hMul (Inv.inv ifp_ …
      nextConts_b_ineq : LE.le (↑(Nat.fib (HAdd.hAdd n 2))) (HAdd.hAdd pred_conts.b  …
      conts_b_ineq : LE.le (↑(Nat.fib (HAdd.hAdd n 1))) conts.b
      zero_lt_conts_b : LT.lt 0 conts.b
      this : LT.lt 0 (HAdd.hAdd pred_conts.b (HMul.hMul gp.b conts.b))
      ⊢ LT.lt 0 den
    -/
    solve_by_elim [mul_pos]
    /-
      🎉 no goals
    -/
  · -- we can cancel multiplication by `conts.b` and addition with `pred_conts.b`
    suffices gp.b * conts.b ≤ ifp_n.fr⁻¹ * conts.b from
      (mul_le_mul_left zero_lt_conts_b).2 <| (add_le_add_iff_left pred_conts.b).2 this
    /-
      case intro.intro.intro.intro.intro.intro.right
      K : Type u_1
      v : K
      n : Nat
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      not_terminatedAt_n : Not ((GenContFract.of v).TerminatedAt n)
      g : GenContFract K := GenContFract.of v
      nextConts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 2)
      conts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
      conts_eq : Eq conts (g.contsAux (HAdd.hAdd n 1))
      pred_conts : GenContFract.Pair K := g.contsAux n
      pred_conts_eq : Eq pred_conts (g.contsAux n)
      gp : GenContFract.Pair K
      s_nth_eq : Eq (g.s.get? n) (Option.some gp)
      gp_a_eq_one : Eq gp.a 1
      nextConts_b_eq : Eq nextConts.b (HAdd.hAdd pred_conts.b (HMul.hMul gp.b conts. …
      den : K := HMul.hMul conts.b (HAdd.hAdd pred_conts.b (HMul.hMul gp.b conts.b))
      ifp_succ_n : GenContFract.IntFractPair K
      succ_nth_stream_eq : Eq (GenContFract.IntFractPair.stream v (HAdd.hAdd n 1)) ( …
      ifp_succ_n_b_eq_gp_b : Eq (↑ifp_succ_n.b) gp.b
      ifp_n : GenContFract.IntFractPair K
      stream_nth_eq : Eq (GenContFract.IntFractPair.stream v n) (Option.some ifp_n)
      stream_nth_fr_ne_zero : Ne ifp_n.fr 0
      if_of_eq_ifp_succ_n : Eq (GenContFract.IntFractPair.of (Inv.inv ifp_n.fr)) ifp …
      den' : K := HMul.hMul conts.b (HAdd.hAdd pred_conts.b (HMul.hMul (Inv.inv ifp_ …
      nextConts_b_ineq : LE.le (↑(Nat.fib (HAdd.hAdd n 2))) (HAdd.hAdd pred_conts.b  …
      conts_b_ineq : LE.le (↑(Nat.fib (HAdd.hAdd n 1))) conts.b
      zero_lt_conts_b : LT.lt 0 conts.b
      ⊢ LE.le (HMul.hMul gp.b conts.b) (HMul.hMul (Inv.inv ifp_n.fr) conts.b)
    -/
    suffices (ifp_succ_n.b : K) * conts.b ≤ ifp_n.fr⁻¹ * conts.b by rwa [← ifp_succ_n_b_eq_gp_b]
    have : (ifp_succ_n.b : K) ≤ ifp_n.fr⁻¹ :=
      IntFractPair.succ_nth_stream_b_le_nth_stream_fr_inv stream_nth_eq succ_nth_stream_eq
    /-
      case intro.intro.intro.intro.intro.intro.right
      K : Type u_1
      v : K
      n : Nat
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      not_terminatedAt_n : Not ((GenContFract.of v).TerminatedAt n)
      g : GenContFract K := GenContFract.of v
      nextConts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 2)
      conts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
      conts_eq : Eq conts (g.contsAux (HAdd.hAdd n 1))
      pred_conts : GenContFract.Pair K := g.contsAux n
      pred_conts_eq : Eq pred_conts (g.contsAux n)
      gp : GenContFract.Pair K
      s_nth_eq : Eq (g.s.get? n) (Option.some gp)
      gp_a_eq_one : Eq gp.a 1
      nextConts_b_eq : Eq nextConts.b (HAdd.hAdd pred_conts.b (HMul.hMul gp.b conts. …
      den : K := HMul.hMul conts.b (HAdd.hAdd pred_conts.b (HMul.hMul gp.b conts.b))
      ifp_succ_n : GenContFract.IntFractPair K
      succ_nth_stream_eq : Eq (GenContFract.IntFractPair.stream v (HAdd.hAdd n 1)) ( …
      ifp_succ_n_b_eq_gp_b : Eq (↑ifp_succ_n.b) gp.b
      ifp_n : GenContFract.IntFractPair K
      stream_nth_eq : Eq (GenContFract.IntFractPair.stream v n) (Option.some ifp_n)
      stream_nth_fr_ne_zero : Ne ifp_n.fr 0
      if_of_eq_ifp_succ_n : Eq (GenContFract.IntFractPair.of (Inv.inv ifp_n.fr)) ifp …
      den' : K := HMul.hMul conts.b (HAdd.hAdd pred_conts.b (HMul.hMul (Inv.inv ifp_ …
      nextConts_b_ineq : LE.le (↑(Nat.fib (HAdd.hAdd n 2))) (HAdd.hAdd pred_conts.b  …
      conts_b_ineq : LE.le (↑(Nat.fib (HAdd.hAdd n 1))) conts.b
      zero_lt_conts_b : LT.lt 0 conts.b
      this : LE.le (↑ifp_succ_n.b) (Inv.inv ifp_n.fr)
      ⊢ LE.le (HMul.hMul (↑ifp_succ_n.b) conts.b) (HMul.hMul (Inv.inv ifp_n.fr) cont …
    -/
    have : 0 ≤ conts.b := le_of_lt zero_lt_conts_b
    /-
      case intro.intro.intro.intro.intro.intro.right
      K : Type u_1
      v : K
      n : Nat
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      not_terminatedAt_n : Not ((GenContFract.of v).TerminatedAt n)
      g : GenContFract K := GenContFract.of v
      nextConts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 2)
      conts : GenContFract.Pair K := g.contsAux (HAdd.hAdd n 1)
      conts_eq : Eq conts (g.contsAux (HAdd.hAdd n 1))
      pred_conts : GenContFract.Pair K := g.contsAux n
      pred_conts_eq : Eq pred_conts (g.contsAux n)
      gp : GenContFract.Pair K
      s_nth_eq : Eq (g.s.get? n) (Option.some gp)
      gp_a_eq_one : Eq gp.a 1
      nextConts_b_eq : Eq nextConts.b (HAdd.hAdd pred_conts.b (HMul.hMul gp.b conts. …
      den : K := HMul.hMul conts.b (HAdd.hAdd pred_conts.b (HMul.hMul gp.b conts.b))
      ifp_succ_n : GenContFract.IntFractPair K
      succ_nth_stream_eq : Eq (GenContFract.IntFractPair.stream v (HAdd.hAdd n 1)) ( …
      ifp_succ_n_b_eq_gp_b : Eq (↑ifp_succ_n.b) gp.b
      ifp_n : GenContFract.IntFractPair K
      stream_nth_eq : Eq (GenContFract.IntFractPair.stream v n) (Option.some ifp_n)
      stream_nth_fr_ne_zero : Ne ifp_n.fr 0
      if_of_eq_ifp_succ_n : Eq (GenContFract.IntFractPair.of (Inv.inv ifp_n.fr)) ifp …
      den' : K := HMul.hMul conts.b (HAdd.hAdd pred_conts.b (HMul.hMul (Inv.inv ifp_ …
      nextConts_b_ineq : LE.le (↑(Nat.fib (HAdd.hAdd n 2))) (HAdd.hAdd pred_conts.b  …
      conts_b_ineq : LE.le (↑(Nat.fib (HAdd.hAdd n 1))) conts.b
      zero_lt_conts_b : LT.lt 0 conts.b
      this✝ : LE.le (↑ifp_succ_n.b) (Inv.inv ifp_n.fr)
      this : LE.le 0 conts.b
      ⊢ LE.le (HMul.hMul (↑ifp_succ_n.b) conts.b) (HMul.hMul (Inv.inv ifp_n.fr) cont …
    -/
    gcongr; exact this
            /-
              🎉 no goals
            -/


/-- Shows that `|v - Aₙ / Bₙ| ≤ 1 / (bₙ * Bₙ * Bₙ)`. This bound is worse than the one shown in
`GenContFract.abs_sub_convs_le`, but sometimes it is easier to apply and
sufficient for one's use case.
 -/
theorem abs_sub_convergents_le' {b : K}
    (nth_partDen_eq : (of v).partDens.get? n = some b) :
    |v - (of v).convs n| ≤ 1 / (b * (of v).dens n * (of v).dens n) := by
  have not_terminatedAt_n : ¬(of v).TerminatedAt n := by
    simp [terminatedAt_iff_partDen_none, nth_partDen_eq]
  /-
    K : Type u_1
    v : K
    n : Nat
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    b : K
    nth_partDen_eq : Eq ((GenContFract.of v).partDens.get? n) (Option.some b)
    not_terminatedAt_n : Not ((GenContFract.of v).TerminatedAt n)
    ⊢ LE.le (abs (HSub.hSub v ((GenContFract.of v).convs n))) (HDiv.hDiv 1 (HMul.h …
  -/
  refine (abs_sub_convs_le not_terminatedAt_n).trans ?_
  -- One can show that `0 < (GenContFract.of v).dens n` but it's easier
  -- to consider the case `(GenContFract.of v).dens n = 0`.
  rcases (zero_le_of_den (K := K)).eq_or_gt with
    ((hB : (GenContFract.of v).dens n = 0) | hB)
    /-
      case inl
      K : Type u_1
      v : K
      n : Nat
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      b : K
      nth_partDen_eq : Eq ((GenContFract.of v).partDens.get? n) (Option.some b)
      not_terminatedAt_n : Not ((GenContFract.of v).TerminatedAt n)
      hB : Eq ((GenContFract.of v).dens n) 0
      ⊢ LE.le (HDiv.hDiv 1 (HMul.hMul ((GenContFract.of v).dens n) ((GenContFract.of …
    -/
  · simp only [hB, mul_zero, zero_mul, div_zero, le_refl]
    /-
      🎉 no goals
    -/
    /-
      case inr
      K : Type u_1
      v : K
      n : Nat
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      b : K
      nth_partDen_eq : Eq ((GenContFract.of v).partDens.get? n) (Option.some b)
      not_terminatedAt_n : Not ((GenContFract.of v).TerminatedAt n)
      hB : LT.lt 0 ((GenContFract.of v).dens n)
      ⊢ LE.le (HDiv.hDiv 1 (HMul.hMul ((GenContFract.of v).dens n) ((GenContFract.of …
    -/
  · apply one_div_le_one_div_of_le
      /-
        case inr.ha
        K : Type u_1
        v : K
        n : Nat
        inst✝¹ : LinearOrderedField K
        inst✝ : FloorRing K
        b : K
        nth_partDen_eq : Eq ((GenContFract.of v).partDens.get? n) (Option.some b)
        not_terminatedAt_n : Not ((GenContFract.of v).TerminatedAt n)
        hB : LT.lt 0 ((GenContFract.of v).dens n)
        ⊢ LT.lt 0 (HMul.hMul (HMul.hMul b ((GenContFract.of v).dens n)) ((GenContFract …
      -/
    · have : 0 < b := zero_lt_one.trans_le (of_one_le_get?_partDen nth_partDen_eq)
      /-
        case inr.ha
        K : Type u_1
        v : K
        n : Nat
        inst✝¹ : LinearOrderedField K
        inst✝ : FloorRing K
        b : K
        nth_partDen_eq : Eq ((GenContFract.of v).partDens.get? n) (Option.some b)
        not_terminatedAt_n : Not ((GenContFract.of v).TerminatedAt n)
        hB : LT.lt 0 ((GenContFract.of v).dens n)
        this : LT.lt 0 b
        ⊢ LT.lt 0 (HMul.hMul (HMul.hMul b ((GenContFract.of v).dens n)) ((GenContFract …
      -/
      apply_rules [mul_pos]
      /-
        🎉 no goals
      -/
      /-
        case inr.h
        K : Type u_1
        v : K
        n : Nat
        inst✝¹ : LinearOrderedField K
        inst✝ : FloorRing K
        b : K
        nth_partDen_eq : Eq ((GenContFract.of v).partDens.get? n) (Option.some b)
        not_terminatedAt_n : Not ((GenContFract.of v).TerminatedAt n)
        hB : LT.lt 0 ((GenContFract.of v).dens n)
        ⊢ LE.le (HMul.hMul (HMul.hMul b ((GenContFract.of v).dens n)) ((GenContFract.o …
      -/
    · conv_rhs => rw [mul_comm]
      /-
        case inr.h
        K : Type u_1
        v : K
        n : Nat
        inst✝¹ : LinearOrderedField K
        inst✝ : FloorRing K
        b : K
        nth_partDen_eq : Eq ((GenContFract.of v).partDens.get? n) (Option.some b)
        not_terminatedAt_n : Not ((GenContFract.of v).TerminatedAt n)
        hB : LT.lt 0 ((GenContFract.of v).dens n)
        ⊢ LE.le (HMul.hMul (HMul.hMul b ((GenContFract.of v).dens n)) ((GenContFract.o …
      -/
      exact mul_le_mul_of_nonneg_right (le_of_succ_get?_den nth_partDen_eq) hB.le
      /-
        🎉 no goals
      -/


