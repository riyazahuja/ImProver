attribute [local simp] Pair.map IntFractPair.mapFr


nonrec theorem exists_gcf_pair_rat_eq_of_nth_contsAux :
    ∃ conts : Pair ℚ, (of v).contsAux n = (conts.map (↑) : Pair K) :=
  Nat.strong_induction_on n
    (by
      /-
        K : Type u_1
        inst✝¹ : LinearOrderedField K
        inst✝ : FloorRing K
        v : K
        n : Nat
        ⊢ ∀ (n : Nat), (∀ (m : Nat), LT.lt m n → Exists fun conts => Eq ((GenContFract …
      -/
      clear n
      /-
        K : Type u_1
        inst✝¹ : LinearOrderedField K
        inst✝ : FloorRing K
        v : K
        ⊢ ∀ (n : Nat), (∀ (m : Nat), LT.lt m n → Exists fun conts => Eq ((GenContFract …
      -/
      let g := of v
      /-
        K : Type u_1
        inst✝¹ : LinearOrderedField K
        inst✝ : FloorRing K
        v : K
        g : GenContFract K := GenContFract.of v
        ⊢ ∀ (n : Nat), (∀ (m : Nat), LT.lt m n → Exists fun conts => Eq ((GenContFract …
      -/
      intro n IH
      /-
        K : Type u_1
        inst✝¹ : LinearOrderedField K
        inst✝ : FloorRing K
        v : K
        g : GenContFract K := GenContFract.of v
        n : Nat
        IH : ∀ (m : Nat), LT.lt m n → Exists fun conts => Eq ((GenContFract.of v).cont …
        ⊢ Exists fun conts => Eq ((GenContFract.of v).contsAux n) (GenContFract.Pair.m …
      -/
      rcases n with (_ | _ | n)
      -- n = 0
        /-
          case zero
          K : Type u_1
          inst✝¹ : LinearOrderedField K
          inst✝ : FloorRing K
          v : K
          g : GenContFract K := GenContFract.of v
          IH : ∀ (m : Nat), LT.lt m 0 → Exists fun conts => Eq ((GenContFract.of v).cont …
          ⊢ Exists fun conts => Eq ((GenContFract.of v).contsAux 0) (GenContFract.Pair.m …
        -/
      · suffices ∃ gp : Pair ℚ, Pair.mk (1 : K) 0 = gp.map (↑) by simpa [contsAux]
        /-
          case zero
          K : Type u_1
          inst✝¹ : LinearOrderedField K
          inst✝ : FloorRing K
          v : K
          g : GenContFract K := GenContFract.of v
          IH : ∀ (m : Nat), LT.lt m 0 → Exists fun conts => Eq ((GenContFract.of v).cont …
          ⊢ Exists fun gp => Eq { a := 1, b := 0 } (GenContFract.Pair.map Rat.cast gp)
        -/
        use Pair.mk 1 0
        /-
          case h
          K : Type u_1
          inst✝¹ : LinearOrderedField K
          inst✝ : FloorRing K
          v : K
          g : GenContFract K := GenContFract.of v
          IH : ∀ (m : Nat), LT.lt m 0 → Exists fun conts => Eq ((GenContFract.of v).cont …
          ⊢ Eq { a := 1, b := 0 } (GenContFract.Pair.map Rat.cast { a := 1, b := 0 })
        -/
        simp
        /-
          🎉 no goals
        -/
      -- n = 1
        /-
          case succ.zero
          K : Type u_1
          inst✝¹ : LinearOrderedField K
          inst✝ : FloorRing K
          v : K
          g : GenContFract K := GenContFract.of v
          IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd 0 1) → Exists fun conts => Eq ((GenContFr …
          ⊢ Exists fun conts => Eq ((GenContFract.of v).contsAux (HAdd.hAdd 0 1)) (GenCo …
        -/
      · suffices ∃ conts : Pair ℚ, Pair.mk g.h 1 = conts.map (↑) by simpa [contsAux]
        /-
          case succ.zero
          K : Type u_1
          inst✝¹ : LinearOrderedField K
          inst✝ : FloorRing K
          v : K
          g : GenContFract K := GenContFract.of v
          IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd 0 1) → Exists fun conts => Eq ((GenContFr …
          ⊢ Exists fun conts => Eq { a := g.h, b := 1 } (GenContFract.Pair.map Rat.cast  …
        -/
        use Pair.mk ⌊v⌋ 1
        /-
          case h
          K : Type u_1
          inst✝¹ : LinearOrderedField K
          inst✝ : FloorRing K
          v : K
          g : GenContFract K := GenContFract.of v
          IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd 0 1) → Exists fun conts => Eq ((GenContFr …
          ⊢ Eq { a := g.h, b := 1 } (GenContFract.Pair.map Rat.cast { a := ↑(Int.floor v …
        -/
        simp [g]
        /-
          🎉 no goals
        -/
      -- 2 ≤ n
        /-
          case succ.succ
          K : Type u_1
          inst✝¹ : LinearOrderedField K
          inst✝ : FloorRing K
          v : K
          g : GenContFract K := GenContFract.of v
          n : Nat
          IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd n 1) 1) → Exists fun conts =>  …
          ⊢ Exists fun conts => Eq ((GenContFract.of v).contsAux (HAdd.hAdd (HAdd.hAdd n …
        -/
      · obtain ⟨pred_conts, pred_conts_eq⟩ := IH (n + 1) <| lt_add_one (n + 1)
        -- invoke the IH
        /-
          case succ.succ.intro
          K : Type u_1
          inst✝¹ : LinearOrderedField K
          inst✝ : FloorRing K
          v : K
          g : GenContFract K := GenContFract.of v
          n : Nat
          IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd n 1) 1) → Exists fun conts =>  …
          pred_conts : GenContFract.Pair Rat
          pred_conts_eq : Eq ((GenContFract.of v).contsAux (HAdd.hAdd n 1)) (GenContFrac …
          ⊢ Exists fun conts => Eq ((GenContFract.of v).contsAux (HAdd.hAdd (HAdd.hAdd n …
        -/
        rcases s_ppred_nth_eq : g.s.get? n with gp_n | gp_n
        -- option.none
          /-
            case succ.succ.intro.none
            K : Type u_1
            inst✝¹ : LinearOrderedField K
            inst✝ : FloorRing K
            v : K
            g : GenContFract K := GenContFract.of v
            n : Nat
            IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd n 1) 1) → Exists fun conts =>  …
            pred_conts : GenContFract.Pair Rat
            pred_conts_eq : Eq ((GenContFract.of v).contsAux (HAdd.hAdd n 1)) (GenContFrac …
            s_ppred_nth_eq : Eq (g.s.get? n) Option.none
            ⊢ Exists fun conts => Eq ((GenContFract.of v).contsAux (HAdd.hAdd (HAdd.hAdd n …
          -/
        · use pred_conts
          have : g.contsAux (n + 2) = g.contsAux (n + 1) :=
            contsAux_stable_of_terminated (n + 1).le_succ s_ppred_nth_eq
          /-
            case h
            K : Type u_1
            inst✝¹ : LinearOrderedField K
            inst✝ : FloorRing K
            v : K
            g : GenContFract K := GenContFract.of v
            n : Nat
            IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd n 1) 1) → Exists fun conts =>  …
            pred_conts : GenContFract.Pair Rat
            pred_conts_eq : Eq ((GenContFract.of v).contsAux (HAdd.hAdd n 1)) (GenContFrac …
            s_ppred_nth_eq : Eq (g.s.get? n) Option.none
            this : Eq (g.contsAux (HAdd.hAdd n 2)) (g.contsAux (HAdd.hAdd n 1))
            ⊢ Eq ((GenContFract.of v).contsAux (HAdd.hAdd (HAdd.hAdd n 1) 1)) (GenContFrac …
          -/
          simp only [g, this, pred_conts_eq]
          /-
            🎉 no goals
          -/
        -- option.some
        · -- invoke the IH a second time
          obtain ⟨ppred_conts, ppred_conts_eq⟩ :=
            IH n <| lt_of_le_of_lt n.le_succ <| lt_add_one <| n + 1
          obtain ⟨a_eq_one, z, b_eq_z⟩ : gp_n.a = 1 ∧ ∃ z : ℤ, gp_n.b = (z : K) :=
            of_partNum_eq_one_and_exists_int_partDen_eq s_ppred_nth_eq
          -- finally, unfold the recurrence to obtain the required rational value.
          simp only [g, a_eq_one, b_eq_z,
            contsAux_recurrence s_ppred_nth_eq ppred_conts_eq pred_conts_eq]
          /-
            case succ.succ.intro.some.intro.intro.intro
            K : Type u_1
            inst✝¹ : LinearOrderedField K
            inst✝ : FloorRing K
            v : K
            g : GenContFract K := GenContFract.of v
            n : Nat
            IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd n 1) 1) → Exists fun conts =>  …
            pred_conts : GenContFract.Pair Rat
            pred_conts_eq : Eq ((GenContFract.of v).contsAux (HAdd.hAdd n 1)) (GenContFrac …
            gp_n : GenContFract.Pair K
            s_ppred_nth_eq : Eq (g.s.get? n) (Option.some gp_n)
            ppred_conts : GenContFract.Pair Rat
            ppred_conts_eq : Eq ((GenContFract.of v).contsAux n) (GenContFract.Pair.map Ra …
            a_eq_one : Eq gp_n.a 1
            z : Int
            b_eq_z : Eq gp_n.b ↑z
            ⊢ Exists fun conts => Eq { a := HAdd.hAdd (HMul.hMul (↑z) (GenContFract.Pair.m …
          -/
          use nextConts 1 (z : ℚ) ppred_conts pred_conts
          /-
            case h
            K : Type u_1
            inst✝¹ : LinearOrderedField K
            inst✝ : FloorRing K
            v : K
            g : GenContFract K := GenContFract.of v
            n : Nat
            IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd n 1) 1) → Exists fun conts =>  …
            pred_conts : GenContFract.Pair Rat
            pred_conts_eq : Eq ((GenContFract.of v).contsAux (HAdd.hAdd n 1)) (GenContFrac …
            gp_n : GenContFract.Pair K
            s_ppred_nth_eq : Eq (g.s.get? n) (Option.some gp_n)
            ppred_conts : GenContFract.Pair Rat
            ppred_conts_eq : Eq ((GenContFract.of v).contsAux n) (GenContFract.Pair.map Ra …
            a_eq_one : Eq gp_n.a 1
            z : Int
            b_eq_z : Eq gp_n.b ↑z
            ⊢ Eq { a := HAdd.hAdd (HMul.hMul (↑z) (GenContFract.Pair.map Rat.cast pred_con …
          -/
          cases ppred_conts; cases pred_conts
          /-
            case h.mk.mk
            K : Type u_1
            inst✝¹ : LinearOrderedField K
            inst✝ : FloorRing K
            v : K
            g : GenContFract K := GenContFract.of v
            n : Nat
            IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd n 1) 1) → Exists fun conts =>  …
            gp_n : GenContFract.Pair K
            s_ppred_nth_eq : Eq (g.s.get? n) (Option.some gp_n)
            a_eq_one : Eq gp_n.a 1
            z : Int
            b_eq_z : Eq gp_n.b ↑z
            a✝¹ b✝¹ : Rat
            ppred_conts_eq : Eq ((GenContFract.of v).contsAux n) (GenContFract.Pair.map Ra …
            a✝ b✝ : Rat
            pred_conts_eq : Eq ((GenContFract.of v).contsAux (HAdd.hAdd n 1)) (GenContFrac …
            ⊢ Eq { a := HAdd.hAdd (HMul.hMul (↑z) (GenContFract.Pair.map Rat.cast { a := a …
          -/
          simp [nextConts, nextNum, nextDen])
          /-
            🎉 no goals
          -/


theorem exists_gcf_pair_rat_eq_nth_conts :
    ∃ conts : Pair ℚ, (of v).conts n = (conts.map (↑) : Pair K) := by
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    n : Nat
    ⊢ Exists fun conts => Eq ((GenContFract.of v).conts n) (GenContFract.Pair.map  …
  -/
  rw [nth_cont_eq_succ_nth_contAux]; exact exists_gcf_pair_rat_eq_of_nth_contsAux v <| n + 1
                                     /-
                                       🎉 no goals
                                     -/


theorem exists_rat_eq_nth_num : ∃ q : ℚ, (of v).nums n = (q : K) := by
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    n : Nat
    ⊢ Exists fun q => Eq ((GenContFract.of v).nums n) ↑q
  -/
  rcases exists_gcf_pair_rat_eq_nth_conts v n with ⟨⟨a, _⟩, nth_cont_eq⟩
  /-
    case intro.mk
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    n : Nat
    a b✝ : Rat
    nth_cont_eq : Eq ((GenContFract.of v).conts n) (GenContFract.Pair.map Rat.cast …
    ⊢ Exists fun q => Eq ((GenContFract.of v).nums n) ↑q
  -/
  use a
  /-
    case h
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    n : Nat
    a b✝ : Rat
    nth_cont_eq : Eq ((GenContFract.of v).conts n) (GenContFract.Pair.map Rat.cast …
    ⊢ Eq ((GenContFract.of v).nums n) ↑a
  -/
  simp [num_eq_conts_a, nth_cont_eq]
  /-
    🎉 no goals
  -/


theorem exists_rat_eq_nth_den : ∃ q : ℚ, (of v).dens n = (q : K) := by
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    n : Nat
    ⊢ Exists fun q => Eq ((GenContFract.of v).dens n) ↑q
  -/
  rcases exists_gcf_pair_rat_eq_nth_conts v n with ⟨⟨_, b⟩, nth_cont_eq⟩
  /-
    case intro.mk
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    n : Nat
    a✝ b : Rat
    nth_cont_eq : Eq ((GenContFract.of v).conts n) (GenContFract.Pair.map Rat.cast …
    ⊢ Exists fun q => Eq ((GenContFract.of v).dens n) ↑q
  -/
  use b
  /-
    case h
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    n : Nat
    a✝ b : Rat
    nth_cont_eq : Eq ((GenContFract.of v).conts n) (GenContFract.Pair.map Rat.cast …
    ⊢ Eq ((GenContFract.of v).dens n) ↑b
  -/
  simp [den_eq_conts_b, nth_cont_eq]
  /-
    🎉 no goals
  -/


/-- Every finite convergent corresponds to a rational number. -/
theorem exists_rat_eq_nth_conv : ∃ q : ℚ, (of v).convs n = (q : K) := by
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    n : Nat
    ⊢ Exists fun q => Eq ((GenContFract.of v).convs n) ↑q
  -/
  rcases exists_rat_eq_nth_num v n with ⟨Aₙ, nth_num_eq⟩
  /-
    case intro
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    n : Nat
    Aₙ : Rat
    nth_num_eq : Eq ((GenContFract.of v).nums n) ↑Aₙ
    ⊢ Exists fun q => Eq ((GenContFract.of v).convs n) ↑q
  -/
  rcases exists_rat_eq_nth_den v n with ⟨Bₙ, nth_den_eq⟩
  /-
    case intro.intro
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    n : Nat
    Aₙ : Rat
    nth_num_eq : Eq ((GenContFract.of v).nums n) ↑Aₙ
    Bₙ : Rat
    nth_den_eq : Eq ((GenContFract.of v).dens n) ↑Bₙ
    ⊢ Exists fun q => Eq ((GenContFract.of v).convs n) ↑q
  -/
  use Aₙ / Bₙ
  /-
    case h
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    n : Nat
    Aₙ : Rat
    nth_num_eq : Eq ((GenContFract.of v).nums n) ↑Aₙ
    Bₙ : Rat
    nth_den_eq : Eq ((GenContFract.of v).dens n) ↑Bₙ
    ⊢ Eq ((GenContFract.of v).convs n) ↑(HDiv.hDiv Aₙ Bₙ)
  -/
  simp [nth_num_eq, nth_den_eq, conv_eq_num_div_den]
  /-
    🎉 no goals
  -/


/-- Every terminating continued fraction corresponds to a rational number. -/
theorem exists_rat_eq_of_terminates (terminates : (of v).Terminates) : ∃ q : ℚ, v = ↑q := by
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    terminates : (GenContFract.of v).Terminates
    ⊢ Exists fun q => Eq v ↑q
  -/
  obtain ⟨n, v_eq_conv⟩ : ∃ n, v = (of v).convs n := of_correctness_of_terminates terminates
  /-
    case intro
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    terminates : (GenContFract.of v).Terminates
    n : Nat
    v_eq_conv : Eq v ((GenContFract.of v).convs n)
    ⊢ Exists fun q => Eq v ↑q
  -/
  obtain ⟨q, conv_eq_q⟩ : ∃ q : ℚ, (of v).convs n = (↑q : K) := exists_rat_eq_nth_conv v n
  /-
    case intro.intro
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    terminates : (GenContFract.of v).Terminates
    n : Nat
    v_eq_conv : Eq v ((GenContFract.of v).convs n)
    q : Rat
    conv_eq_q : Eq ((GenContFract.of v).convs n) ↑q
    ⊢ Exists fun q => Eq v ↑q
  -/
  have : v = (↑q : K) := Eq.trans v_eq_conv conv_eq_q
  /-
    case intro.intro
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    terminates : (GenContFract.of v).Terminates
    n : Nat
    v_eq_conv : Eq v ((GenContFract.of v).convs n)
    q : Rat
    conv_eq_q : Eq ((GenContFract.of v).convs n) ↑q
    this : Eq v ↑q
    ⊢ Exists fun q => Eq v ↑q
  -/
  use q, this
  /-
    🎉 no goals
  -/


theorem coe_of_rat_eq (v_eq_q : v = (↑q : K)) :
    ((IntFractPair.of q).mapFr (↑) : IntFractPair K) = IntFractPair.of v := by
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    q : Rat
    v_eq_q : Eq v ↑q
    ⊢ Eq (GenContFract.IntFractPair.mapFr Rat.cast (GenContFract.IntFractPair.of q …
  -/
  simp [IntFractPair.of, v_eq_q]
  /-
    🎉 no goals
  -/


theorem coe_stream_nth_rat_eq (v_eq_q : v = (↑q : K)) (n : ℕ) :
    ((IntFractPair.stream q n).map (mapFr (↑)) : Option <| IntFractPair K) =
      IntFractPair.stream v n := by
  induction n with
  | zero =>
    -- Porting note: was
    -- simp [IntFractPair.stream, coe_of_rat_eq v_eq_q]
    simp only [IntFractPair.stream, Option.map_some', coe_of_rat_eq v_eq_q]
  | succ n IH =>
    rw [v_eq_q] at IH
    cases stream_q_nth_eq : IntFractPair.stream q n with
    | none => simp [IntFractPair.stream, IH.symm, v_eq_q, stream_q_nth_eq]
    | some ifp_n =>
      obtain ⟨b, fr⟩ := ifp_n
      rcases Decidable.em (fr = 0) with fr_zero | fr_ne_zero
      · simp [IntFractPair.stream, IH.symm, v_eq_q, stream_q_nth_eq, fr_zero]
      · replace IH : some (IntFractPair.mk b (fr : K)) = IntFractPair.stream (↑q) n := by
          rwa [stream_q_nth_eq] at IH
        have : (fr : K)⁻¹ = ((fr⁻¹ : ℚ) : K) := by norm_cast
        have coe_of_fr := coe_of_rat_eq this
        simpa [IntFractPair.stream, IH.symm, v_eq_q, stream_q_nth_eq, fr_ne_zero]


theorem coe_stream'_rat_eq (v_eq_q : v = (↑q : K)) :
    ((IntFractPair.stream q).map (Option.map (mapFr (↑))) : Stream' <| Option <| IntFractPair K) =
      IntFractPair.stream v := by
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    q : Rat
    v_eq_q : Eq v ↑q
    ⊢ Eq (Stream'.map (Option.map (GenContFract.IntFractPair.mapFr Rat.cast)) (Gen …
  -/
  funext n; exact IntFractPair.coe_stream_nth_rat_eq v_eq_q n
            /-
              🎉 no goals
            -/


theorem coe_of_h_rat_eq (v_eq_q : v = (↑q : K)) : (↑((of q).h : ℚ) : K) = (of v).h := by
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    q : Rat
    v_eq_q : Eq v ↑q
    ⊢ Eq (↑(GenContFract.of q).h) (GenContFract.of v).h
  -/
  unfold of IntFractPair.seq1
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    q : Rat
    v_eq_q : Eq v ↑q
    ⊢ Eq (↑(GenContFract.of.match_1 (fun x => GenContFract Rat) { fst := GenContFr …
  -/
  rw [← IntFractPair.coe_of_rat_eq v_eq_q]
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    q : Rat
    v_eq_q : Eq v ↑q
    ⊢ Eq (↑(GenContFract.of.match_1 (fun x => GenContFract Rat) { fst := GenContFr …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem coe_of_s_get?_rat_eq (v_eq_q : v = (↑q : K)) (n : ℕ) :
    (((of q).s.get? n).map (Pair.map (↑)) : Option <| Pair K) = (of v).s.get? n := by
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    q : Rat
    v_eq_q : Eq v ↑q
    n : Nat
    ⊢ Eq (Option.map (GenContFract.Pair.map Rat.cast) ((GenContFract.of q).s.get?  …
  -/
  simp only [of, IntFractPair.seq1, Stream'.Seq.map_get?, Stream'.Seq.get?_tail]
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    q : Rat
    v_eq_q : Eq v ↑q
    n : Nat
    ⊢ Eq (Option.map (GenContFract.Pair.map Rat.cast) (Option.map (fun p => { a := …
  -/
  simp only [Stream'.Seq.get?]
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    q : Rat
    v_eq_q : Eq v ↑q
    n : Nat
    ⊢ Eq (Option.map (GenContFract.Pair.map Rat.cast) (Option.map (fun p => { a := …
  -/
  rw [← IntFractPair.coe_stream'_rat_eq v_eq_q]
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    q : Rat
    v_eq_q : Eq v ↑q
    n : Nat
    ⊢ Eq (Option.map (GenContFract.Pair.map Rat.cast) (Option.map (fun p => { a := …
  -/
  rcases succ_nth_stream_eq : IntFractPair.stream q (n + 1) with (_ | ⟨_, _⟩) <;>
    /-
      case none
      K : Type u_1
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      v : K
      q : Rat
      v_eq_q : Eq v ↑q
      n : Nat
      succ_nth_stream_eq : Eq (GenContFract.IntFractPair.stream q (HAdd.hAdd n 1)) O …
      ⊢ Eq (Option.map (GenContFract.Pair.map Rat.cast) (Option.map (fun p => { a := …
    -/
    /-
      🎉 no goals
    -/
    simp [Stream'.map, Stream'.get, succ_nth_stream_eq]
    /-
      🎉 no goals
    -/


theorem coe_of_s_rat_eq (v_eq_q : v = (↑q : K)) :
    ((of q).s.map (Pair.map ((↑))) : Stream'.Seq <| Pair K) = (of v).s := by
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    q : Rat
    v_eq_q : Eq v ↑q
    ⊢ Eq (Stream'.Seq.map (GenContFract.Pair.map Rat.cast) (GenContFract.of q).s)  …
  -/
  ext n; rw [← coe_of_s_get?_rat_eq v_eq_q]; rfl
                                             /-
                                               🎉 no goals
                                             -/


/-- Given `(v : K), (q : ℚ), and v = q`, we have that `of q = of v` -/
theorem coe_of_rat_eq (v_eq_q : v = (↑q : K)) :
    (⟨(of q).h, (of q).s.map (Pair.map (↑))⟩ : GenContFract K) = of v := by
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    q : Rat
    v_eq_q : Eq v ↑q
    ⊢ Eq { h := ↑(GenContFract.of q).h, s := Stream'.Seq.map (GenContFract.Pair.ma …
  -/
  rcases gcf_v_eq : of v with ⟨h, s⟩; subst v
  -- Porting note: made coercion target explicit
  /-
    case mk
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    q : Rat
    h : K
    s : Stream'.Seq (GenContFract.Pair K)
    gcf_v_eq : Eq (GenContFract.of ↑q) { h := h, s := s }
    ⊢ Eq { h := ↑(GenContFract.of q).h, s := Stream'.Seq.map (GenContFract.Pair.ma …
  -/
  obtain rfl : ↑⌊(q : K)⌋ = h := by injection gcf_v_eq
  -- Porting note: was
  -- simp [coe_of_h_rat_eq rfl, coe_of_s_rat_eq rfl, gcf_v_eq]
  simp only [gcf_v_eq, Int.cast_inj, Rat.floor_cast, of_h_eq_floor, eq_self_iff_true,
    Rat.cast_intCast, and_self, coe_of_h_rat_eq rfl, coe_of_s_rat_eq rfl]


theorem of_terminates_iff_of_rat_terminates {v : K} {q : ℚ} (v_eq_q : v = (q : K)) :
    (of v).Terminates ↔ (of q).Terminates := by
  /-
    K : Type u_1
    inst✝¹ : LinearOrderedField K
    inst✝ : FloorRing K
    v : K
    q : Rat
    v_eq_q : Eq v ↑q
    ⊢ Iff (GenContFract.of v).Terminates (GenContFract.of q).Terminates
  -/
  constructor <;> intro h <;> obtain ⟨n, h⟩ := h <;> use n <;>
    /-
      case h
      K : Type u_1
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      v : K
      q : Rat
      v_eq_q : Eq v ↑q
      n : Nat
      h : (GenContFract.of v).s.TerminatedAt n
      ⊢ (GenContFract.of q).s.TerminatedAt n
    -/
    simp only [Stream'.Seq.TerminatedAt, (coe_of_s_get?_rat_eq v_eq_q n).symm] at h ⊢ <;>
    /-
      case h
      K : Type u_1
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      v : K
      q : Rat
      v_eq_q : Eq v ↑q
      n : Nat
      h : Eq (Option.map (GenContFract.Pair.map Rat.cast) ((GenContFract.of q).s.get …
      ⊢ Eq ((GenContFract.of q).s.get? n) Option.none
    -/
    cases h' : (of q).s.get? n <;>
    /-
      case h.none
      K : Type u_1
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      v : K
      q : Rat
      v_eq_q : Eq v ↑q
      n : Nat
      h : Eq (Option.map (GenContFract.Pair.map Rat.cast) ((GenContFract.of q).s.get …
      h' : Eq ((GenContFract.of q).s.get? n) Option.none
      ⊢ Eq Option.none Option.none
    -/
    simp only [h'] at h <;> -- Porting note: added
    /-
      case h.none
      K : Type u_1
      inst✝¹ : LinearOrderedField K
      inst✝ : FloorRing K
      v : K
      q : Rat
      v_eq_q : Eq v ↑q
      n : Nat
      h' : Eq ((GenContFract.of q).s.get? n) Option.none
      h : Eq (Option.map (GenContFract.Pair.map Rat.cast) Option.none) Option.none
      ⊢ Eq Option.none Option.none
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
    trivial
    /-
      🎉 no goals
    -/


/-- Shows that for any `q : ℚ` with `0 < q < 1`, the numerator of the fractional part of
`IntFractPair.of q⁻¹` is smaller than the numerator of `q`.
-/
theorem of_inv_fr_num_lt_num_of_pos (q_pos : 0 < q) : (IntFractPair.of q⁻¹).fr.num < q.num :=
  Rat.fract_inv_num_lt_num_of_pos q_pos


/-- Shows that the sequence of numerators of the fractional parts of the stream is strictly
antitone. -/
theorem stream_succ_nth_fr_num_lt_nth_fr_num_rat {ifp_n ifp_succ_n : IntFractPair ℚ}
    (stream_nth_eq : IntFractPair.stream q n = some ifp_n)
    (stream_succ_nth_eq : IntFractPair.stream q (n + 1) = some ifp_succ_n) :
    ifp_succ_n.fr.num < ifp_n.fr.num := by
  obtain ⟨ifp_n', stream_nth_eq', ifp_n_fract_ne_zero, IntFractPair.of_eq_ifp_succ_n⟩ :
    ∃ ifp_n',
      IntFractPair.stream q n = some ifp_n' ∧
        ifp_n'.fr ≠ 0 ∧ IntFractPair.of ifp_n'.fr⁻¹ = ifp_succ_n :=
    succ_nth_stream_eq_some_iff.mp stream_succ_nth_eq
  /-
    case intro.intro.intro
    q : Rat
    n : Nat
    ifp_n ifp_succ_n : GenContFract.IntFractPair Rat
    stream_nth_eq : Eq (GenContFract.IntFractPair.stream q n) (Option.some ifp_n)
    stream_succ_nth_eq : Eq (GenContFract.IntFractPair.stream q (HAdd.hAdd n 1)) ( …
    ifp_n' : GenContFract.IntFractPair Rat
    stream_nth_eq' : Eq (GenContFract.IntFractPair.stream q n) (Option.some ifp_n')
    ifp_n_fract_ne_zero : Ne ifp_n'.fr 0
    IntFractPair.of_eq_ifp_succ_n : Eq (GenContFract.IntFractPair.of (Inv.inv ifp_ …
    ⊢ LT.lt ifp_succ_n.fr.num ifp_n.fr.num
  -/
  have : ifp_n = ifp_n' := by injection Eq.trans stream_nth_eq.symm stream_nth_eq'
  /-
    case intro.intro.intro
    q : Rat
    n : Nat
    ifp_n ifp_succ_n : GenContFract.IntFractPair Rat
    stream_nth_eq : Eq (GenContFract.IntFractPair.stream q n) (Option.some ifp_n)
    stream_succ_nth_eq : Eq (GenContFract.IntFractPair.stream q (HAdd.hAdd n 1)) ( …
    ifp_n' : GenContFract.IntFractPair Rat
    stream_nth_eq' : Eq (GenContFract.IntFractPair.stream q n) (Option.some ifp_n')
    ifp_n_fract_ne_zero : Ne ifp_n'.fr 0
    IntFractPair.of_eq_ifp_succ_n : Eq (GenContFract.IntFractPair.of (Inv.inv ifp_ …
    this : Eq ifp_n ifp_n'
    ⊢ LT.lt ifp_succ_n.fr.num ifp_n.fr.num
  -/
  cases this
  /-
    case intro.intro.intro.refl
    q : Rat
    n : Nat
    ifp_n ifp_succ_n : GenContFract.IntFractPair Rat
    stream_nth_eq : Eq (GenContFract.IntFractPair.stream q n) (Option.some ifp_n)
    stream_succ_nth_eq : Eq (GenContFract.IntFractPair.stream q (HAdd.hAdd n 1)) ( …
    stream_nth_eq' : Eq (GenContFract.IntFractPair.stream q n) (Option.some ifp_n)
    ifp_n_fract_ne_zero : Ne ifp_n.fr 0
    IntFractPair.of_eq_ifp_succ_n : Eq (GenContFract.IntFractPair.of (Inv.inv ifp_ …
    ⊢ LT.lt ifp_succ_n.fr.num ifp_n.fr.num
  -/
  rw [← IntFractPair.of_eq_ifp_succ_n]
  /-
    case intro.intro.intro.refl
    q : Rat
    n : Nat
    ifp_n ifp_succ_n : GenContFract.IntFractPair Rat
    stream_nth_eq : Eq (GenContFract.IntFractPair.stream q n) (Option.some ifp_n)
    stream_succ_nth_eq : Eq (GenContFract.IntFractPair.stream q (HAdd.hAdd n 1)) ( …
    stream_nth_eq' : Eq (GenContFract.IntFractPair.stream q n) (Option.some ifp_n)
    ifp_n_fract_ne_zero : Ne ifp_n.fr 0
    IntFractPair.of_eq_ifp_succ_n : Eq (GenContFract.IntFractPair.of (Inv.inv ifp_ …
    ⊢ LT.lt (GenContFract.IntFractPair.of (Inv.inv ifp_n.fr)).fr.num ifp_n.fr.num
  -/
  obtain ⟨zero_le_ifp_n_fract, _⟩ := nth_stream_fr_nonneg_lt_one stream_nth_eq
  /-
    case intro.intro.intro.refl.intro
    q : Rat
    n : Nat
    ifp_n ifp_succ_n : GenContFract.IntFractPair Rat
    stream_nth_eq : Eq (GenContFract.IntFractPair.stream q n) (Option.some ifp_n)
    stream_succ_nth_eq : Eq (GenContFract.IntFractPair.stream q (HAdd.hAdd n 1)) ( …
    stream_nth_eq' : Eq (GenContFract.IntFractPair.stream q n) (Option.some ifp_n)
    ifp_n_fract_ne_zero : Ne ifp_n.fr 0
    IntFractPair.of_eq_ifp_succ_n : Eq (GenContFract.IntFractPair.of (Inv.inv ifp_ …
    zero_le_ifp_n_fract : LE.le 0 ifp_n.fr
    right✝ : LT.lt ifp_n.fr 1
    ⊢ LT.lt (GenContFract.IntFractPair.of (Inv.inv ifp_n.fr)).fr.num ifp_n.fr.num
  -/
  have : 0 < ifp_n.fr := lt_of_le_of_ne zero_le_ifp_n_fract <| ifp_n_fract_ne_zero.symm
  /-
    case intro.intro.intro.refl.intro
    q : Rat
    n : Nat
    ifp_n ifp_succ_n : GenContFract.IntFractPair Rat
    stream_nth_eq : Eq (GenContFract.IntFractPair.stream q n) (Option.some ifp_n)
    stream_succ_nth_eq : Eq (GenContFract.IntFractPair.stream q (HAdd.hAdd n 1)) ( …
    stream_nth_eq' : Eq (GenContFract.IntFractPair.stream q n) (Option.some ifp_n)
    ifp_n_fract_ne_zero : Ne ifp_n.fr 0
    IntFractPair.of_eq_ifp_succ_n : Eq (GenContFract.IntFractPair.of (Inv.inv ifp_ …
    zero_le_ifp_n_fract : LE.le 0 ifp_n.fr
    right✝ : LT.lt ifp_n.fr 1
    this : LT.lt 0 ifp_n.fr
    ⊢ LT.lt (GenContFract.IntFractPair.of (Inv.inv ifp_n.fr)).fr.num ifp_n.fr.num
  -/
  exact of_inv_fr_num_lt_num_of_pos this
  /-
    🎉 no goals
  -/


theorem stream_nth_fr_num_le_fr_num_sub_n_rat :
    ∀ {ifp_n : IntFractPair ℚ},
      IntFractPair.stream q n = some ifp_n → ifp_n.fr.num ≤ (IntFractPair.of q).fr.num - n := by
  induction n with
  | zero =>
    intro ifp_zero stream_zero_eq
    have : IntFractPair.of q = ifp_zero := by injection stream_zero_eq
    simp [le_refl, this.symm]
  | succ n IH =>
    intro ifp_succ_n stream_succ_nth_eq
    suffices ifp_succ_n.fr.num + 1 ≤ (IntFractPair.of q).fr.num - n by
      rw [Int.ofNat_succ, sub_add_eq_sub_sub]
      solve_by_elim [le_sub_right_of_add_le]
    rcases succ_nth_stream_eq_some_iff.mp stream_succ_nth_eq with ⟨ifp_n, stream_nth_eq, -⟩
    have : ifp_succ_n.fr.num < ifp_n.fr.num :=
      stream_succ_nth_fr_num_lt_nth_fr_num_rat stream_nth_eq stream_succ_nth_eq
    have : ifp_succ_n.fr.num + 1 ≤ ifp_n.fr.num := Int.add_one_le_of_lt this
    exact le_trans this (IH stream_nth_eq)


theorem exists_nth_stream_eq_none_of_rat (q : ℚ) : ∃ n : ℕ, IntFractPair.stream q n = none := by
  /-
    q : Rat
    ⊢ Exists fun n => Eq (GenContFract.IntFractPair.stream q n) Option.none
  -/
  let fract_q_num := (Int.fract q).num; let n := fract_q_num.natAbs + 1
  /-
    q : Rat
    fract_q_num : Int := (Int.fract q).num
    n : Nat := HAdd.hAdd fract_q_num.natAbs 1
    ⊢ Exists fun n => Eq (GenContFract.IntFractPair.stream q n) Option.none
  -/
  rcases stream_nth_eq : IntFractPair.stream q n with ifp | ifp
    /-
      case none
      q : Rat
      fract_q_num : Int := (Int.fract q).num
      n : Nat := HAdd.hAdd fract_q_num.natAbs 1
      stream_nth_eq : Eq (GenContFract.IntFractPair.stream q n) Option.none
      ⊢ Exists fun n => Eq (GenContFract.IntFractPair.stream q n) Option.none
    -/
  · use n, stream_nth_eq
    /-
      🎉 no goals
    -/
  · -- arrive at a contradiction since the numerator decreased num + 1 times but every fractional
    -- value is nonnegative.
    have ifp_fr_num_le_q_fr_num_sub_n : ifp.fr.num ≤ fract_q_num - n :=
      stream_nth_fr_num_le_fr_num_sub_n_rat stream_nth_eq
    have : fract_q_num - n = -1 := by
      have : 0 ≤ fract_q_num := Rat.num_nonneg.mpr (Int.fract_nonneg q)
      -- Porting note: was
      -- simp [Int.natAbs_of_nonneg this, sub_add_eq_sub_sub_swap, sub_right_comm]
      simp only [n, Nat.cast_add, Int.natAbs_of_nonneg this, Nat.cast_one,
        sub_add_eq_sub_sub_swap, sub_right_comm, sub_self, zero_sub]
    /-
      case some
      q : Rat
      fract_q_num : Int := (Int.fract q).num
      n : Nat := HAdd.hAdd fract_q_num.natAbs 1
      ifp : GenContFract.IntFractPair Rat
      stream_nth_eq : Eq (GenContFract.IntFractPair.stream q n) (Option.some ifp)
      ifp_fr_num_le_q_fr_num_sub_n : LE.le ifp.fr.num (HSub.hSub fract_q_num ↑n)
      this : Eq (HSub.hSub fract_q_num ↑n) (-1)
      ⊢ Exists fun n => Eq (GenContFract.IntFractPair.stream q n) Option.none
    -/
    have : 0 ≤ ifp.fr := (nth_stream_fr_nonneg_lt_one stream_nth_eq).left
    /-
      case some
      q : Rat
      fract_q_num : Int := (Int.fract q).num
      n : Nat := HAdd.hAdd fract_q_num.natAbs 1
      ifp : GenContFract.IntFractPair Rat
      stream_nth_eq : Eq (GenContFract.IntFractPair.stream q n) (Option.some ifp)
      ifp_fr_num_le_q_fr_num_sub_n : LE.le ifp.fr.num (HSub.hSub fract_q_num ↑n)
      this✝ : Eq (HSub.hSub fract_q_num ↑n) (-1)
      this : LE.le 0 ifp.fr
      ⊢ Exists fun n => Eq (GenContFract.IntFractPair.stream q n) Option.none
    -/
    have : 0 ≤ ifp.fr.num := Rat.num_nonneg.mpr this
    /-
      case some
      q : Rat
      fract_q_num : Int := (Int.fract q).num
      n : Nat := HAdd.hAdd fract_q_num.natAbs 1
      ifp : GenContFract.IntFractPair Rat
      stream_nth_eq : Eq (GenContFract.IntFractPair.stream q n) (Option.some ifp)
      ifp_fr_num_le_q_fr_num_sub_n : LE.le ifp.fr.num (HSub.hSub fract_q_num ↑n)
      this✝¹ : Eq (HSub.hSub fract_q_num ↑n) (-1)
      this✝ : LE.le 0 ifp.fr
      this : LE.le 0 ifp.fr.num
      ⊢ Exists fun n => Eq (GenContFract.IntFractPair.stream q n) Option.none
    -/
    omega
    /-
      🎉 no goals
    -/


/-- The continued fraction of a rational number terminates. -/
theorem terminates_of_rat (q : ℚ) : (of q).Terminates :=
  Exists.elim (IntFractPair.exists_nth_stream_eq_none_of_rat q) fun n stream_nth_eq_none =>
    Exists.intro n
      (have : IntFractPair.stream q (n + 1) = none := IntFractPair.stream_isSeq q stream_nth_eq_none
      of_terminatedAt_n_iff_succ_nth_intFractPair_stream_eq_none.mpr this)


/-- The continued fraction `GenContFract.of v` terminates if and only if `v ∈ ℚ`. -/
theorem terminates_iff_rat (v : K) : (of v).Terminates ↔ ∃ q : ℚ, v = (q : K) :=
  Iff.intro
    (fun terminates_v : (of v).Terminates =>
      show ∃ q : ℚ, v = (q : K) from exists_rat_eq_of_terminates terminates_v)
    fun exists_q_eq_v : ∃ q : ℚ, v = (↑q : K) =>
    Exists.elim exists_q_eq_v fun q => fun v_eq_q : v = ↑q =>
      have : (of q).Terminates := terminates_of_rat q
      (of_terminates_iff_of_rat_terminates v_eq_q).mpr this


