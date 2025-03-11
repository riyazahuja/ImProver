theorem stream_zero (v : K) : IntFractPair.stream v 0 = some (IntFractPair.of v) :=
  rfl


theorem stream_eq_none_of_fr_eq_zero {ifp_n : IntFractPair K}
    (stream_nth_eq : IntFractPair.stream v n = some ifp_n) (nth_fr_eq_zero : ifp_n.fr = 0) :
    IntFractPair.stream v (n + 1) = none := by
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    n : Nat
    ifp_n : GenContFract.IntFractPair K
    stream_nth_eq : Eq (GenContFract.IntFractPair.stream v n) (Option.some ifp_n)
    nth_fr_eq_zero : Eq ifp_n.fr 0
    ⊢ Eq (GenContFract.IntFractPair.stream v (HAdd.hAdd n 1)) Option.none
  -/
  obtain ⟨_, fr⟩ := ifp_n
  /-
    case mk
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    n : Nat
    b✝ : Int
    fr : K
    stream_nth_eq : Eq (GenContFract.IntFractPair.stream v n) (Option.some { b :=  …
    nth_fr_eq_zero : Eq { b := b✝, fr := fr }.fr 0
    ⊢ Eq (GenContFract.IntFractPair.stream v (HAdd.hAdd n 1)) Option.none
  -/
  change fr = 0 at nth_fr_eq_zero
  /-
    case mk
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    n : Nat
    b✝ : Int
    fr : K
    stream_nth_eq : Eq (GenContFract.IntFractPair.stream v n) (Option.some { b :=  …
    nth_fr_eq_zero : Eq fr 0
    ⊢ Eq (GenContFract.IntFractPair.stream v (HAdd.hAdd n 1)) Option.none
  -/
  simp [IntFractPair.stream, stream_nth_eq, nth_fr_eq_zero]
  /-
    🎉 no goals
  -/


/-- Gives a recurrence to compute the `n + 1`th value of the sequence of integer and fractional
parts of a value in case of termination.
-/
theorem succ_nth_stream_eq_none_iff :
    IntFractPair.stream v (n + 1) = none ↔
      IntFractPair.stream v n = none ∨ ∃ ifp, IntFractPair.stream v n = some ifp ∧ ifp.fr = 0 := by
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    n : Nat
    ⊢ Iff (Eq (GenContFract.IntFractPair.stream v (HAdd.hAdd n 1)) Option.none) (O …
  -/
  rw [IntFractPair.stream]
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    n : Nat
    ⊢ Iff (Eq ((GenContFract.IntFractPair.stream v n).bind fun ap_n => ite (Eq ap_ …
  -/
                                    /-
                                      🎉 no goals
                                    -/
  cases IntFractPair.stream v n <;> simp [imp_false]
                                    /-
                                      🎉 no goals
                                    -/


/-- Gives a recurrence to compute the `n + 1`th value of the sequence of integer and fractional
parts of a value in case of non-termination.
-/
theorem succ_nth_stream_eq_some_iff {ifp_succ_n : IntFractPair K} :
    IntFractPair.stream v (n + 1) = some ifp_succ_n ↔
      ∃ ifp_n : IntFractPair K,
        IntFractPair.stream v n = some ifp_n ∧
          ifp_n.fr ≠ 0 ∧ IntFractPair.of ifp_n.fr⁻¹ = ifp_succ_n := by
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    n : Nat
    ifp_succ_n : GenContFract.IntFractPair K
    ⊢ Iff (Eq (GenContFract.IntFractPair.stream v (HAdd.hAdd n 1)) (Option.some if …
  -/
  simp [IntFractPair.stream, ite_eq_iff, Option.bind_eq_some]
  /-
    🎉 no goals
  -/


/-- An easier to use version of one direction of
`GenContFract.IntFractPair.succ_nth_stream_eq_some_iff`. -/
theorem stream_succ_of_some {p : IntFractPair K} (h : IntFractPair.stream v n = some p)
    (h' : p.fr ≠ 0) : IntFractPair.stream v (n + 1) = some (IntFractPair.of p.fr⁻¹) :=
  succ_nth_stream_eq_some_iff.mpr ⟨p, h, h', rfl⟩


/-- The stream of `IntFractPair`s of an integer stops after the first term.
-/
theorem stream_succ_of_int (a : ℤ) (n : ℕ) : IntFractPair.stream (a : K) (n + 1) = none := by
  induction n with
  | zero =>
    refine IntFractPair.stream_eq_none_of_fr_eq_zero (IntFractPair.stream_zero (a : K)) ?_
    simp only [IntFractPair.of, Int.fract_intCast]
  | succ n ih => exact IntFractPair.succ_nth_stream_eq_none_iff.mpr (Or.inl ih)


theorem exists_succ_nth_stream_of_fr_zero {ifp_succ_n : IntFractPair K}
    (stream_succ_nth_eq : IntFractPair.stream v (n + 1) = some ifp_succ_n)
    (succ_nth_fr_eq_zero : ifp_succ_n.fr = 0) :
    ∃ ifp_n : IntFractPair K, IntFractPair.stream v n = some ifp_n ∧ ifp_n.fr⁻¹ = ⌊ifp_n.fr⁻¹⌋ := by
  -- get the witness from `succ_nth_stream_eq_some_iff` and prove that it has the additional
  -- properties
  rcases succ_nth_stream_eq_some_iff.mp stream_succ_nth_eq with
    ⟨ifp_n, seq_nth_eq, _, rfl⟩
  /-
    case intro.intro.intro
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    n : Nat
    ifp_n : GenContFract.IntFractPair K
    seq_nth_eq : Eq (GenContFract.IntFractPair.stream v n) (Option.some ifp_n)
    left✝ : Ne ifp_n.fr 0
    stream_succ_nth_eq : Eq (GenContFract.IntFractPair.stream v (HAdd.hAdd n 1)) ( …
    succ_nth_fr_eq_zero : Eq (GenContFract.IntFractPair.of (Inv.inv ifp_n.fr)).fr 0
    ⊢ Exists fun ifp_n => And (Eq (GenContFract.IntFractPair.stream v n) (Option.s …
  -/
  refine ⟨ifp_n, seq_nth_eq, ?_⟩
  /-
    case intro.intro.intro
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    n : Nat
    ifp_n : GenContFract.IntFractPair K
    seq_nth_eq : Eq (GenContFract.IntFractPair.stream v n) (Option.some ifp_n)
    left✝ : Ne ifp_n.fr 0
    stream_succ_nth_eq : Eq (GenContFract.IntFractPair.stream v (HAdd.hAdd n 1)) ( …
    succ_nth_fr_eq_zero : Eq (GenContFract.IntFractPair.of (Inv.inv ifp_n.fr)).fr 0
    ⊢ Eq (Inv.inv ifp_n.fr) ↑(Int.floor (Inv.inv ifp_n.fr))
  -/
  simpa only [IntFractPair.of, Int.fract, sub_eq_zero] using succ_nth_fr_eq_zero
  /-
    🎉 no goals
  -/


/-- A recurrence relation that expresses the `(n+1)`th term of the stream of `IntFractPair`s
of `v` for non-integer `v` in terms of the `n`th term of the stream associated to
the inverse of the fractional part of `v`.
-/
theorem stream_succ (h : Int.fract v ≠ 0) (n : ℕ) :
    IntFractPair.stream v (n + 1) = IntFractPair.stream (Int.fract v)⁻¹ n := by
  induction n with
  | zero =>
    have H : (IntFractPair.of v).fr = Int.fract v := by simp [IntFractPair.of]
    rw [stream_zero, stream_succ_of_some (stream_zero v) (ne_of_eq_of_ne H h), H]
  | succ n ih =>
    rcases eq_or_ne (IntFractPair.stream (Int.fract v)⁻¹ n) none with hnone | hsome
    · rw [hnone] at ih
      rw [succ_nth_stream_eq_none_iff.mpr (Or.inl hnone),
        succ_nth_stream_eq_none_iff.mpr (Or.inl ih)]
    · obtain ⟨p, hp⟩ := Option.ne_none_iff_exists'.mp hsome
      rw [hp] at ih
      rcases eq_or_ne p.fr 0 with hz | hnz
      · rw [stream_eq_none_of_fr_eq_zero hp hz, stream_eq_none_of_fr_eq_zero ih hz]
      · rw [stream_succ_of_some hp hnz, stream_succ_of_some ih hnz]


/-- The head term of the sequence with head of `v` is just the integer part of `v`. -/
@[simp]
theorem IntFractPair.seq1_fst_eq_of : (IntFractPair.seq1 v).fst = IntFractPair.of v :=
  rfl


theorem of_h_eq_intFractPair_seq1_fst_b : (of v).h = (IntFractPair.seq1 v).fst.b := by
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    ⊢ Eq (GenContFract.of v).h ↑(GenContFract.IntFractPair.seq1 v).1.b
  -/
  cases aux_seq_eq : IntFractPair.seq1 v
  /-
    case mk
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    fst✝ : GenContFract.IntFractPair K
    snd✝ : Stream'.Seq (GenContFract.IntFractPair K)
    aux_seq_eq : Eq (GenContFract.IntFractPair.seq1 v) { fst := fst✝, snd := snd✝ }
    ⊢ Eq (GenContFract.of v).h ↑{ fst := fst✝, snd := snd✝ }.1.b
  -/
  simp [of, aux_seq_eq]
  /-
    🎉 no goals
  -/


/-- The head term of the gcf of `v` is `⌊v⌋`. -/
@[simp]
theorem of_h_eq_floor : (of v).h = ⌊v⌋ := by
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    ⊢ Eq (GenContFract.of v).h ↑(Int.floor v)
  -/
  simp [of_h_eq_intFractPair_seq1_fst_b, IntFractPair.of]
  /-
    🎉 no goals
  -/


theorem IntFractPair.get?_seq1_eq_succ_get?_stream :
    (IntFractPair.seq1 v).snd.get? n = (IntFractPair.stream v) (n + 1) :=
  rfl


theorem of_terminatedAt_iff_intFractPair_seq1_terminatedAt :
    (of v).TerminatedAt n ↔ (IntFractPair.seq1 v).snd.TerminatedAt n :=
  Option.map_eq_none


theorem of_terminatedAt_n_iff_succ_nth_intFractPair_stream_eq_none :
    (of v).TerminatedAt n ↔ IntFractPair.stream v (n + 1) = none := by
  rw [of_terminatedAt_iff_intFractPair_seq1_terminatedAt, Stream'.Seq.TerminatedAt,
    IntFractPair.get?_seq1_eq_succ_get?_stream]


theorem IntFractPair.exists_succ_get?_stream_of_gcf_of_get?_eq_some {gp_n : Pair K}
    (s_nth_eq : (of v).s.get? n = some gp_n) :
    ∃ ifp : IntFractPair K, IntFractPair.stream v (n + 1) = some ifp ∧ (ifp.b : K) = gp_n.b := by
  obtain ⟨ifp, stream_succ_nth_eq, gp_n_eq⟩ :
    ∃ ifp, IntFractPair.stream v (n + 1) = some ifp ∧ Pair.mk 1 (ifp.b : K) = gp_n := by
    unfold of IntFractPair.seq1 at s_nth_eq
    simpa [Stream'.Seq.get?_tail, Stream'.Seq.map_get?] using s_nth_eq
  /-
    case intro.intro
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    n : Nat
    gp_n : GenContFract.Pair K
    s_nth_eq : Eq ((GenContFract.of v).s.get? n) (Option.some gp_n)
    ifp : GenContFract.IntFractPair K
    stream_succ_nth_eq : Eq (GenContFract.IntFractPair.stream v (HAdd.hAdd n 1)) ( …
    gp_n_eq : Eq { a := 1, b := ↑ifp.b } gp_n
    ⊢ Exists fun ifp => And (Eq (GenContFract.IntFractPair.stream v (HAdd.hAdd n 1 …
  -/
  cases gp_n_eq
  /-
    case intro.intro.refl
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    n : Nat
    ifp : GenContFract.IntFractPair K
    stream_succ_nth_eq : Eq (GenContFract.IntFractPair.stream v (HAdd.hAdd n 1)) ( …
    s_nth_eq : Eq ((GenContFract.of v).s.get? n) (Option.some { a := 1, b := ↑ifp. …
    ⊢ Exists fun ifp_1 => And (Eq (GenContFract.IntFractPair.stream v (HAdd.hAdd n …
  -/
  simp_all only [Option.some.injEq, exists_eq_left']
  /-
    🎉 no goals
  -/


/-- Shows how the entries of the sequence of the computed continued fraction can be obtained by the
integer parts of the stream of integer and fractional parts.
-/
theorem get?_of_eq_some_of_succ_get?_intFractPair_stream {ifp_succ_n : IntFractPair K}
    (stream_succ_nth_eq : IntFractPair.stream v (n + 1) = some ifp_succ_n) :
    (of v).s.get? n = some ⟨1, ifp_succ_n.b⟩ := by
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    n : Nat
    ifp_succ_n : GenContFract.IntFractPair K
    stream_succ_nth_eq : Eq (GenContFract.IntFractPair.stream v (HAdd.hAdd n 1)) ( …
    ⊢ Eq ((GenContFract.of v).s.get? n) (Option.some { a := 1, b := ↑ifp_succ_n.b })
  -/
  unfold of IntFractPair.seq1
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    n : Nat
    ifp_succ_n : GenContFract.IntFractPair K
    stream_succ_nth_eq : Eq (GenContFract.IntFractPair.stream v (HAdd.hAdd n 1)) ( …
    ⊢ Eq ((GenContFract.of.match_1 (fun x => GenContFract K) { fst := GenContFract …
  -/
  simp [Stream'.Seq.map_tail, Stream'.Seq.get?_tail, Stream'.Seq.map_get?, stream_succ_nth_eq]
  /-
    🎉 no goals
  -/


/-- Shows how the entries of the sequence of the computed continued fraction can be obtained by the
fractional parts of the stream of integer and fractional parts.
-/
theorem get?_of_eq_some_of_get?_intFractPair_stream_fr_ne_zero {ifp_n : IntFractPair K}
    (stream_nth_eq : IntFractPair.stream v n = some ifp_n) (nth_fr_ne_zero : ifp_n.fr ≠ 0) :
    (of v).s.get? n = some ⟨1, (IntFractPair.of ifp_n.fr⁻¹).b⟩ :=
  have : IntFractPair.stream v (n + 1) = some (IntFractPair.of ifp_n.fr⁻¹) := by
    /-
      K : Type u_1
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      v : K
      n : Nat
      ifp_n : GenContFract.IntFractPair K
      stream_nth_eq : Eq (GenContFract.IntFractPair.stream v n) (Option.some ifp_n)
      nth_fr_ne_zero : Ne ifp_n.fr 0
      ⊢ Eq (GenContFract.IntFractPair.stream v (HAdd.hAdd n 1)) (Option.some (GenCon …
    -/
    cases ifp_n
    simp only [IntFractPair.stream, Nat.add_eq, add_zero, stream_nth_eq, Option.some_bind,
      ite_eq_right_iff]
    /-
      case mk
      K : Type u_1
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      v : K
      n : Nat
      b✝ : Int
      fr✝ : K
      stream_nth_eq : Eq (GenContFract.IntFractPair.stream v n) (Option.some { b :=  …
      nth_fr_ne_zero : Ne { b := b✝, fr := fr✝ }.fr 0
      ⊢ Eq fr✝ 0 → Eq Option.none (Option.some (GenContFract.IntFractPair.of (Inv.in …
    -/
    intro; contradiction
           /-
             🎉 no goals
           -/
  get?_of_eq_some_of_succ_get?_intFractPair_stream this


theorem of_s_head_aux (v : K) : (of v).s.get? 0 = (IntFractPair.stream v 1).bind (some ∘ fun p =>
    { a := 1
      b := p.b }) := by
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    ⊢ Eq ((GenContFract.of v).s.get? 0) ((GenContFract.IntFractPair.stream v 1).bi …
  -/
  rw [of, IntFractPair.seq1]
  simp only [of, Stream'.Seq.map_tail, Stream'.Seq.map, Stream'.Seq.tail, Stream'.Seq.head,
    Stream'.Seq.get?, Stream'.map]
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    ⊢ Eq (Option.map (fun p => { a := 1, b := ↑p.b }) ((GenContFract.IntFractPair. …
  -/
  rw [← Stream'.get_succ, Stream'.get, Option.map.eq_def]
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    ⊢ Eq (Option.getD.match_1 (fun x => Option (GenContFract.Pair K)) (GenContFrac …
  -/
            /-
              🎉 no goals
            -/
  split <;> simp_all only [Option.some_bind, Option.none_bind, Function.comp_apply]
            /-
              🎉 no goals
            -/


/-- This gives the first pair of coefficients of the continued fraction of a non-integer `v`.
-/
theorem of_s_head (h : fract v ≠ 0) : (of v).s.head = some ⟨1, ⌊(fract v)⁻¹⌋⟩ := by
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    h : Ne (Int.fract v) 0
    ⊢ Eq (GenContFract.of v).s.head (Option.some { a := 1, b := ↑(Int.floor (Inv.i …
  -/
  change (of v).s.get? 0 = _
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    h : Ne (Int.fract v) 0
    ⊢ Eq ((GenContFract.of v).s.get? 0) (Option.some { a := 1, b := ↑(Int.floor (I …
  -/
  rw [of_s_head_aux, stream_succ_of_some (stream_zero v) h, Option.bind]
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    h : Ne (Int.fract v) 0
    ⊢ Eq (Function.comp Option.some (fun p => { a := 1, b := ↑p.b }) (GenContFract …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- If `a` is an integer, then the coefficient sequence of its continued fraction is empty.
-/
theorem of_s_of_int (a : ℤ) : (of (a : K)).s = Stream'.Seq.nil :=
  haveI h : ∀ n, (of (a : K)).s.get? n = none := by
    /-
      K : Type u_1
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      a : Int
      ⊢ ∀ (n : Nat), Eq ((GenContFract.of ↑a).s.get? n) Option.none
    -/
    intro n
    induction n with
    | zero => rw [of_s_head_aux, stream_succ_of_int, Option.bind]
    | succ n ih => exact (of (a : K)).s.prop ih
  Stream'.Seq.ext fun n => (h n).trans (Stream'.Seq.get?_nil n).symm


/-- Recurrence for the `GenContFract.of` an element `v` of `K` in terms of that of the inverse of
the fractional part of `v`.
-/
theorem of_s_succ (n : ℕ) : (of v).s.get? (n + 1) = (of (fract v)⁻¹).s.get? n := by
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    n : Nat
    ⊢ Eq ((GenContFract.of v).s.get? (HAdd.hAdd n 1)) ((GenContFract.of (Inv.inv ( …
  -/
  rcases eq_or_ne (fract v) 0 with h | h
    /-
      case inl
      K : Type u_1
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      v : K
      n : Nat
      h : Eq (Int.fract v) 0
      ⊢ Eq ((GenContFract.of v).s.get? (HAdd.hAdd n 1)) ((GenContFract.of (Inv.inv ( …
    -/
  · obtain ⟨a, rfl⟩ : ∃ a : ℤ, v = a := ⟨⌊v⌋, eq_of_sub_eq_zero h⟩
    rw [fract_intCast, inv_zero, of_s_of_int, ← cast_zero, of_s_of_int,
      Stream'.Seq.get?_nil, Stream'.Seq.get?_nil]
  /-
    case inr
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    n : Nat
    h : Ne (Int.fract v) 0
    ⊢ Eq ((GenContFract.of v).s.get? (HAdd.hAdd n 1)) ((GenContFract.of (Inv.inv ( …
  -/
  rcases eq_or_ne ((of (fract v)⁻¹).s.get? n) none with h₁ | h₁
  · rwa [h₁, ← terminatedAt_iff_s_none,
      of_terminatedAt_n_iff_succ_nth_intFractPair_stream_eq_none, stream_succ h, ←
      of_terminatedAt_n_iff_succ_nth_intFractPair_stream_eq_none, terminatedAt_iff_s_none]
    /-
      case inr.inr
      K : Type u_1
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      v : K
      n : Nat
      h : Ne (Int.fract v) 0
      h₁ : Ne ((GenContFract.of (Inv.inv (Int.fract v))).s.get? n) Option.none
      ⊢ Eq ((GenContFract.of v).s.get? (HAdd.hAdd n 1)) ((GenContFract.of (Inv.inv ( …
    -/
  · obtain ⟨p, hp⟩ := Option.ne_none_iff_exists'.mp h₁
    /-
      case inr.inr.intro
      K : Type u_1
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      v : K
      n : Nat
      h : Ne (Int.fract v) 0
      h₁ : Ne ((GenContFract.of (Inv.inv (Int.fract v))).s.get? n) Option.none
      p : GenContFract.Pair K
      hp : Eq ((GenContFract.of (Inv.inv (Int.fract v))).s.get? n) (Option.some p)
      ⊢ Eq ((GenContFract.of v).s.get? (HAdd.hAdd n 1)) ((GenContFract.of (Inv.inv ( …
    -/
    obtain ⟨p', hp'₁, _⟩ := exists_succ_get?_stream_of_gcf_of_get?_eq_some hp
    /-
      case inr.inr.intro.intro.intro
      K : Type u_1
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      v : K
      n : Nat
      h : Ne (Int.fract v) 0
      h₁ : Ne ((GenContFract.of (Inv.inv (Int.fract v))).s.get? n) Option.none
      p : GenContFract.Pair K
      hp : Eq ((GenContFract.of (Inv.inv (Int.fract v))).s.get? n) (Option.some p)
      p' : GenContFract.IntFractPair K
      hp'₁ : Eq (GenContFract.IntFractPair.stream (Inv.inv (Int.fract v)) (HAdd.hAdd …
      right✝ : Eq (↑p'.b) p.b
      ⊢ Eq ((GenContFract.of v).s.get? (HAdd.hAdd n 1)) ((GenContFract.of (Inv.inv ( …
    -/
    have Hp := get?_of_eq_some_of_succ_get?_intFractPair_stream hp'₁
    /-
      case inr.inr.intro.intro.intro
      K : Type u_1
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      v : K
      n : Nat
      h : Ne (Int.fract v) 0
      h₁ : Ne ((GenContFract.of (Inv.inv (Int.fract v))).s.get? n) Option.none
      p : GenContFract.Pair K
      hp : Eq ((GenContFract.of (Inv.inv (Int.fract v))).s.get? n) (Option.some p)
      p' : GenContFract.IntFractPair K
      hp'₁ : Eq (GenContFract.IntFractPair.stream (Inv.inv (Int.fract v)) (HAdd.hAdd …
      right✝ : Eq (↑p'.b) p.b
      Hp : Eq ((GenContFract.of (Inv.inv (Int.fract v))).s.get? n) (Option.some { a  …
      ⊢ Eq ((GenContFract.of v).s.get? (HAdd.hAdd n 1)) ((GenContFract.of (Inv.inv ( …
    -/
    rw [← stream_succ h] at hp'₁
    /-
      case inr.inr.intro.intro.intro
      K : Type u_1
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      v : K
      n : Nat
      h : Ne (Int.fract v) 0
      h₁ : Ne ((GenContFract.of (Inv.inv (Int.fract v))).s.get? n) Option.none
      p : GenContFract.Pair K
      hp : Eq ((GenContFract.of (Inv.inv (Int.fract v))).s.get? n) (Option.some p)
      p' : GenContFract.IntFractPair K
      hp'₁ : Eq (GenContFract.IntFractPair.stream v (HAdd.hAdd (HAdd.hAdd n 1) 1)) ( …
      right✝ : Eq (↑p'.b) p.b
      Hp : Eq ((GenContFract.of (Inv.inv (Int.fract v))).s.get? n) (Option.some { a  …
      ⊢ Eq ((GenContFract.of v).s.get? (HAdd.hAdd n 1)) ((GenContFract.of (Inv.inv ( …
    -/
    rw [Hp, get?_of_eq_some_of_succ_get?_intFractPair_stream hp'₁]
    /-
      🎉 no goals
    -/


/-- This expresses the tail of the coefficient sequence of the `GenContFract.of` an element `v` of
`K` as the coefficient sequence of that of the inverse of the fractional part of `v`.
-/
theorem of_s_tail : (of v).s.tail = (of (fract v)⁻¹).s :=
  Stream'.Seq.ext fun n => Stream'.Seq.get?_tail (of v).s n ▸ of_s_succ v n


/-- If `a` is an integer, then the `convs'` of its continued fraction expansion
are all equal to `a`.
-/
theorem convs'_of_int (a : ℤ) : (of (a : K)).convs' n = a := by
  induction n with
  | zero => simp only [zeroth_conv'_eq_h, of_h_eq_floor, floor_intCast]
  | succ =>
    rw [convs', of_h_eq_floor, floor_intCast, add_right_eq_self]
    exact convs'Aux_succ_none ((of_s_of_int K a).symm ▸ Stream'.Seq.get?_nil 0) _


/-- The recurrence relation for the `convs'` of the continued fraction expansion
of an element `v` of `K` in terms of the convergents of the inverse of its fractional part.
-/
theorem convs'_succ :
    (of v).convs' (n + 1) = ⌊v⌋ + 1 / (of (fract v)⁻¹).convs' n := by
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    n : Nat
    ⊢ Eq ((GenContFract.of v).convs' (HAdd.hAdd n 1)) (HAdd.hAdd (↑(Int.floor v))  …
  -/
  rcases eq_or_ne (fract v) 0 with h | h
    /-
      case inl
      K : Type u_1
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      v : K
      n : Nat
      h : Eq (Int.fract v) 0
      ⊢ Eq ((GenContFract.of v).convs' (HAdd.hAdd n 1)) (HAdd.hAdd (↑(Int.floor v))  …
    -/
  · obtain ⟨a, rfl⟩ : ∃ a : ℤ, v = a := ⟨⌊v⌋, eq_of_sub_eq_zero h⟩
    rw [convs'_of_int, fract_intCast, inv_zero, ← cast_zero, convs'_of_int, cast_zero,
      div_zero, add_zero, floor_intCast]
    /-
      case inr
      K : Type u_1
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      v : K
      n : Nat
      h : Ne (Int.fract v) 0
      ⊢ Eq ((GenContFract.of v).convs' (HAdd.hAdd n 1)) (HAdd.hAdd (↑(Int.floor v))  …
    -/
  · rw [convs', of_h_eq_floor, add_right_inj, convs'Aux_succ_some (of_s_head h)]
    /-
      case inr
      K : Type u_1
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      v : K
      n : Nat
      h : Ne (Int.fract v) 0
      ⊢ Eq (HDiv.hDiv { a := 1, b := ↑(Int.floor (Inv.inv (Int.fract v))) }.a (HAdd. …
    -/
    exact congr_arg (1 / ·) (by rw [convs', of_h_eq_floor, add_right_inj, of_s_tail])
    /-
      🎉 no goals
    -/


