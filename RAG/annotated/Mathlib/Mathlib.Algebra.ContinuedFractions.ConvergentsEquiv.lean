/-- Given a sequence of `GenContFract.Pair`s `s = [(a₀, bₒ), (a₁, b₁), ...]`, `squashSeq s n`
combines `⟨aₙ, bₙ⟩` and `⟨aₙ₊₁, bₙ₊₁⟩` at position `n` to `⟨aₙ, bₙ + aₙ₊₁ / bₙ₊₁⟩`. For example,
`squashSeq s 0 = [(a₀, bₒ + a₁ / b₁), (a₁, b₁),...]`.
If `s.TerminatedAt (n + 1)`, then `squashSeq s n = s`.
-/
def squashSeq (s : Stream'.Seq <| Pair K) (n : ℕ) : Stream'.Seq (Pair K) :=
  match Prod.mk (s.get? n) (s.get? (n + 1)) with
  | ⟨some gp_n, some gp_succ_n⟩ =>
    Stream'.Seq.nats.zipWith
      -- return the squashed value at position `n`; otherwise, do nothing.
      (fun n' gp => if n' = n then ⟨gp_n.a, gp_n.b + gp_succ_n.a / gp_succ_n.b⟩ else gp) s
  | _ => s


/-- If the sequence already terminated at position `n + 1`, nothing gets squashed. -/
theorem squashSeq_eq_self_of_terminated (terminatedAt_succ_n : s.TerminatedAt (n + 1)) :
    squashSeq s n = s := by
  /-
    K : Type u_1
    n : Nat
    s : Stream'.Seq (GenContFract.Pair K)
    inst✝ : DivisionRing K
    terminatedAt_succ_n : s.TerminatedAt (HAdd.hAdd n 1)
    ⊢ Eq (GenContFract.squashSeq s n) s
  -/
  change s.get? (n + 1) = none at terminatedAt_succ_n
  /-
    K : Type u_1
    n : Nat
    s : Stream'.Seq (GenContFract.Pair K)
    inst✝ : DivisionRing K
    terminatedAt_succ_n : Eq (s.get? (HAdd.hAdd n 1)) Option.none
    ⊢ Eq (GenContFract.squashSeq s n) s
  -/
                                /-
                                  🎉 no goals
                                -/
  cases s_nth_eq : s.get? n <;> simp only [*, squashSeq]
                                /-
                                  🎉 no goals
                                -/


/-- If the sequence has not terminated before position `n + 1`, the value at `n + 1` gets
squashed into position `n`. -/
theorem squashSeq_nth_of_not_terminated {gp_n gp_succ_n : Pair K} (s_nth_eq : s.get? n = some gp_n)
    (s_succ_nth_eq : s.get? (n + 1) = some gp_succ_n) :
    (squashSeq s n).get? n = some ⟨gp_n.a, gp_n.b + gp_succ_n.a / gp_succ_n.b⟩ := by
  /-
    K : Type u_1
    n : Nat
    s : Stream'.Seq (GenContFract.Pair K)
    inst✝ : DivisionRing K
    gp_n gp_succ_n : GenContFract.Pair K
    s_nth_eq : Eq (s.get? n) (Option.some gp_n)
    s_succ_nth_eq : Eq (s.get? (HAdd.hAdd n 1)) (Option.some gp_succ_n)
    ⊢ Eq ((GenContFract.squashSeq s n).get? n) (Option.some { a := gp_n.a, b := HA …
  -/
  simp [*, squashSeq]
  /-
    🎉 no goals
  -/


/-- The values before the squashed position stay the same. -/
theorem squashSeq_nth_of_lt {m : ℕ} (m_lt_n : m < n) : (squashSeq s n).get? m = s.get? m := by
  cases s_succ_nth_eq : s.get? (n + 1) with
  | none => rw [squashSeq_eq_self_of_terminated s_succ_nth_eq]
  | some =>
    obtain ⟨gp_n, s_nth_eq⟩ : ∃ gp_n, s.get? n = some gp_n :=
      s.ge_stable n.le_succ s_succ_nth_eq
    obtain ⟨gp_m, s_mth_eq⟩ : ∃ gp_m, s.get? m = some gp_m :=
      s.ge_stable (le_of_lt m_lt_n) s_nth_eq
    simp [*, squashSeq, m_lt_n.ne]


/-- Squashing at position `n + 1` and taking the tail is the same as squashing the tail of the
sequence at position `n`. -/
theorem squashSeq_succ_n_tail_eq_squashSeq_tail_n :
    (squashSeq s (n + 1)).tail = squashSeq s.tail n := by
  cases s_succ_succ_nth_eq : s.get? (n + 2) with
  | none =>
    cases s_succ_nth_eq : s.get? (n + 1) <;>
      simp only [squashSeq, Stream'.Seq.get?_tail, s_succ_nth_eq, s_succ_succ_nth_eq]
  | some gp_succ_succ_n =>
    obtain ⟨gp_succ_n, s_succ_nth_eq⟩ : ∃ gp_succ_n, s.get? (n + 1) = some gp_succ_n :=
      s.ge_stable (n + 1).le_succ s_succ_succ_nth_eq
    -- apply extensionality with `m` and continue by cases `m = n`.
    ext1 m
    rcases Decidable.em (m = n) with m_eq_n | m_ne_n
    · simp [*, squashSeq]
    · cases s_succ_mth_eq : s.get? (m + 1)
      · simp only [*, squashSeq, Stream'.Seq.get?_tail, Stream'.Seq.get?_zipWith,
          Option.map₂_none_right]
      · simp [*, squashSeq]


/-- The auxiliary function `convs'Aux` returns the same value for a sequence and the
corresponding squashed sequence at the squashed position. -/
theorem succ_succ_nth_conv'Aux_eq_succ_nth_conv'Aux_squashSeq :
    convs'Aux s (n + 2) = convs'Aux (squashSeq s n) (n + 1) := by
  cases s_succ_nth_eq : s.get? <| n + 1 with
  | none =>
    rw [squashSeq_eq_self_of_terminated s_succ_nth_eq,
      convs'Aux_stable_step_of_terminated s_succ_nth_eq]
  | some gp_succ_n =>
    induction n generalizing s gp_succ_n with
    | zero =>
      obtain ⟨gp_head, s_head_eq⟩ : ∃ gp_head, s.head = some gp_head :=
        s.ge_stable zero_le_one s_succ_nth_eq
      have : (squashSeq s 0).head = some ⟨gp_head.a, gp_head.b + gp_succ_n.a / gp_succ_n.b⟩ :=
        squashSeq_nth_of_not_terminated s_head_eq s_succ_nth_eq
      simp_all [convs'Aux, Stream'.Seq.head, Stream'.Seq.get?_tail]
    | succ m IH =>
      obtain ⟨gp_head, s_head_eq⟩ : ∃ gp_head, s.head = some gp_head :=
        s.ge_stable (m + 2).zero_le s_succ_nth_eq
      suffices
        gp_head.a / (gp_head.b + convs'Aux s.tail (m + 2)) =
          convs'Aux (squashSeq s (m + 1)) (m + 2)
        by simpa only [convs'Aux, s_head_eq]
      have : (squashSeq s (m + 1)).head = some gp_head :=
        (squashSeq_nth_of_lt m.succ_pos).trans s_head_eq
      simp_all [convs'Aux, squashSeq_succ_n_tail_eq_squashSeq_tail_n]


/-- Given a gcf `g = [h; (a₀, bₒ), (a₁, b₁), ...]`, we have
- `squashGCF g 0 = [h + a₀ / b₀); (a₀, bₒ), ...]`,
- `squashGCF g (n + 1) = ⟨g.h, squashSeq g.s n⟩`
-/
def squashGCF (g : GenContFract K) : ℕ → GenContFract K
  | 0 =>
    match g.s.get? 0 with
    | none => g
    | some gp => ⟨g.h + gp.a / gp.b, g.s⟩
  | n + 1 => ⟨g.h, squashSeq g.s n⟩


/-- If the gcf already terminated at position `n`, nothing gets squashed. -/
theorem squashGCF_eq_self_of_terminated (terminatedAt_n : TerminatedAt g n) :
    squashGCF g n = g := by
  cases n with
  | zero =>
    change g.s.get? 0 = none at terminatedAt_n
    simp only [convs', squashGCF, convs'Aux, terminatedAt_n]
  | succ =>
    cases g
    simp only [squashGCF, mk.injEq, true_and]
    exact squashSeq_eq_self_of_terminated terminatedAt_n


/-- The values before the squashed position stay the same. -/
theorem squashGCF_nth_of_lt {m : ℕ} (m_lt_n : m < n) :
    (squashGCF g (n + 1)).s.get? m = g.s.get? m := by
  /-
    K : Type u_1
    n : Nat
    g : GenContFract K
    inst✝ : DivisionRing K
    m : Nat
    m_lt_n : LT.lt m n
    ⊢ Eq ((g.squashGCF (HAdd.hAdd n 1)).s.get? m) (g.s.get? m)
  -/
  simp only [squashGCF, squashSeq_nth_of_lt m_lt_n, Nat.add_eq, add_zero]
  /-
    🎉 no goals
  -/


/-- `convs'` returns the same value for a gcf and the corresponding squashed gcf at the
squashed position. -/
theorem succ_nth_conv'_eq_squashGCF_nth_conv' :
    g.convs' (n + 1) = (squashGCF g n).convs' n := by
  cases n with
  | zero =>
    cases g_s_head_eq : g.s.get? 0 <;>
      simp [g_s_head_eq, squashGCF, convs', convs'Aux, Stream'.Seq.head]
  | succ =>
    simp only [succ_succ_nth_conv'Aux_eq_succ_nth_conv'Aux_squashSeq, convs',
      squashGCF]


/-- The auxiliary continuants before the squashed position stay the same. -/
theorem contsAux_eq_contsAux_squashGCF_of_le {m : ℕ} :
    m ≤ n → contsAux g m = (squashGCF g n).contsAux m :=
  Nat.strong_induction_on m
    (by
      /-
        K : Type u_1
        n : Nat
        g : GenContFract K
        inst✝ : DivisionRing K
        m : Nat
        ⊢ ∀ (n_1 : Nat), (∀ (m : Nat), LT.lt m n_1 → LE.le m n → Eq (g.contsAux m) ((g …
      -/
      clear m
      /-
        K : Type u_1
        n : Nat
        g : GenContFract K
        inst✝ : DivisionRing K
        ⊢ ∀ (n_1 : Nat), (∀ (m : Nat), LT.lt m n_1 → LE.le m n → Eq (g.contsAux m) ((g …
      -/
      intro m IH m_le_n
      /-
        K : Type u_1
        n : Nat
        g : GenContFract K
        inst✝ : DivisionRing K
        m : Nat
        IH : ∀ (m_1 : Nat), LT.lt m_1 m → LE.le m_1 n → Eq (g.contsAux m_1) ((g.squash …
        m_le_n : LE.le m n
        ⊢ Eq (g.contsAux m) ((g.squashGCF n).contsAux m)
      -/
      rcases m with - | m'
        /-
          case zero
          K : Type u_1
          n : Nat
          g : GenContFract K
          inst✝ : DivisionRing K
          IH : ∀ (m : Nat), LT.lt m 0 → LE.le m n → Eq (g.contsAux m) ((g.squashGCF n).c …
          m_le_n : LE.le 0 n
          ⊢ Eq (g.contsAux 0) ((g.squashGCF n).contsAux 0)
        -/
      · rfl
        /-
          🎉 no goals
        -/
        /-
          case succ
          K : Type u_1
          n : Nat
          g : GenContFract K
          inst✝ : DivisionRing K
          m' : Nat
          IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd m' 1) → LE.le m n → Eq (g.contsAux m) ((g …
          m_le_n : LE.le (HAdd.hAdd m' 1) n
          ⊢ Eq (g.contsAux (HAdd.hAdd m' 1)) ((g.squashGCF n).contsAux (HAdd.hAdd m' 1))
        -/
      · rcases n with - | n'
          /-
            case succ.zero
            K : Type u_1
            g : GenContFract K
            inst✝ : DivisionRing K
            m' : Nat
            IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd m' 1) → LE.le m 0 → Eq (g.contsAux m) ((g …
            m_le_n : LE.le (HAdd.hAdd m' 1) 0
            ⊢ Eq (g.contsAux (HAdd.hAdd m' 1)) ((g.squashGCF 0).contsAux (HAdd.hAdd m' 1))
          -/
        · exact (m'.not_succ_le_zero m_le_n).elim
          /-
            🎉 no goals
          -/
        -- 1 ≰ 0
          /-
            case succ.succ
            K : Type u_1
            g : GenContFract K
            inst✝ : DivisionRing K
            m' n' : Nat
            IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd m' 1) → LE.le m (HAdd.hAdd n' 1) → Eq (g. …
            m_le_n : LE.le (HAdd.hAdd m' 1) (HAdd.hAdd n' 1)
            ⊢ Eq (g.contsAux (HAdd.hAdd m' 1)) ((g.squashGCF (HAdd.hAdd n' 1)).contsAux (H …
          -/
        · rcases m' with - | m''
            /-
              case succ.succ.zero
              K : Type u_1
              g : GenContFract K
              inst✝ : DivisionRing K
              n' : Nat
              IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd 0 1) → LE.le m (HAdd.hAdd n' 1) → Eq (g.c …
              m_le_n : LE.le (HAdd.hAdd 0 1) (HAdd.hAdd n' 1)
              ⊢ Eq (g.contsAux (HAdd.hAdd 0 1)) ((g.squashGCF (HAdd.hAdd n' 1)).contsAux (HA …
            -/
          · rfl
            /-
              🎉 no goals
            -/
          · -- get some inequalities to instantiate the IH for m'' and m'' + 1
            /-
              case succ.succ.succ
              K : Type u_1
              g : GenContFract K
              inst✝ : DivisionRing K
              n' m'' : Nat
              IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd m'' 1) 1) → LE.le m (HAdd.hAdd …
              m_le_n : LE.le (HAdd.hAdd (HAdd.hAdd m'' 1) 1) (HAdd.hAdd n' 1)
              ⊢ Eq (g.contsAux (HAdd.hAdd (HAdd.hAdd m'' 1) 1)) ((g.squashGCF (HAdd.hAdd n'  …
            -/
            have m'_lt_n : m'' + 1 < n' + 1 := m_le_n
            /-
              case succ.succ.succ
              K : Type u_1
              g : GenContFract K
              inst✝ : DivisionRing K
              n' m'' : Nat
              IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd m'' 1) 1) → LE.le m (HAdd.hAdd …
              m_le_n : LE.le (HAdd.hAdd (HAdd.hAdd m'' 1) 1) (HAdd.hAdd n' 1)
              m'_lt_n : LT.lt (HAdd.hAdd m'' 1) (HAdd.hAdd n' 1)
              ⊢ Eq (g.contsAux (HAdd.hAdd (HAdd.hAdd m'' 1) 1)) ((g.squashGCF (HAdd.hAdd n'  …
            -/
            have succ_m''th_contsAux_eq := IH (m'' + 1) (lt_add_one (m'' + 1)) m'_lt_n.le
            /-
              case succ.succ.succ
              K : Type u_1
              g : GenContFract K
              inst✝ : DivisionRing K
              n' m'' : Nat
              IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd m'' 1) 1) → LE.le m (HAdd.hAdd …
              m_le_n : LE.le (HAdd.hAdd (HAdd.hAdd m'' 1) 1) (HAdd.hAdd n' 1)
              m'_lt_n : LT.lt (HAdd.hAdd m'' 1) (HAdd.hAdd n' 1)
              succ_m''th_contsAux_eq : Eq (g.contsAux (HAdd.hAdd m'' 1)) ((g.squashGCF (HAdd …
              ⊢ Eq (g.contsAux (HAdd.hAdd (HAdd.hAdd m'' 1) 1)) ((g.squashGCF (HAdd.hAdd n'  …
            -/
            have : m'' < m'' + 2 := lt_add_of_pos_right m'' zero_lt_two
            /-
              case succ.succ.succ
              K : Type u_1
              g : GenContFract K
              inst✝ : DivisionRing K
              n' m'' : Nat
              IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd m'' 1) 1) → LE.le m (HAdd.hAdd …
              m_le_n : LE.le (HAdd.hAdd (HAdd.hAdd m'' 1) 1) (HAdd.hAdd n' 1)
              m'_lt_n : LT.lt (HAdd.hAdd m'' 1) (HAdd.hAdd n' 1)
              succ_m''th_contsAux_eq : Eq (g.contsAux (HAdd.hAdd m'' 1)) ((g.squashGCF (HAdd …
              this : LT.lt m'' (HAdd.hAdd m'' 2)
              ⊢ Eq (g.contsAux (HAdd.hAdd (HAdd.hAdd m'' 1) 1)) ((g.squashGCF (HAdd.hAdd n'  …
            -/
            have m''th_contsAux_eq := IH m'' this (le_trans this.le m_le_n)
            have : (squashGCF g (n' + 1)).s.get? m'' = g.s.get? m'' :=
              squashGCF_nth_of_lt (Nat.succ_lt_succ_iff.mp m'_lt_n)
            /-
              case succ.succ.succ
              K : Type u_1
              g : GenContFract K
              inst✝ : DivisionRing K
              n' m'' : Nat
              IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd m'' 1) 1) → LE.le m (HAdd.hAdd …
              m_le_n : LE.le (HAdd.hAdd (HAdd.hAdd m'' 1) 1) (HAdd.hAdd n' 1)
              m'_lt_n : LT.lt (HAdd.hAdd m'' 1) (HAdd.hAdd n' 1)
              succ_m''th_contsAux_eq : Eq (g.contsAux (HAdd.hAdd m'' 1)) ((g.squashGCF (HAdd …
              this✝ : LT.lt m'' (HAdd.hAdd m'' 2)
              m''th_contsAux_eq : Eq (g.contsAux m'') ((g.squashGCF (HAdd.hAdd n' 1)).contsA …
              this : Eq ((g.squashGCF (HAdd.hAdd n' 1)).s.get? m'') (g.s.get? m'')
              ⊢ Eq (g.contsAux (HAdd.hAdd (HAdd.hAdd m'' 1) 1)) ((g.squashGCF (HAdd.hAdd n'  …
            -/
            simp [contsAux, succ_m''th_contsAux_eq, m''th_contsAux_eq, this])
            /-
              🎉 no goals
            -/


/-- The convergents coincide in the expected way at the squashed position if the partial denominator
at the squashed position is not zero. -/
theorem succ_nth_conv_eq_squashGCF_nth_conv [Field K]
    (nth_partDen_ne_zero : ∀ {b : K}, g.partDens.get? n = some b → b ≠ 0) :
    g.convs (n + 1) = (squashGCF g n).convs n := by
  /-
    K : Type u_1
    n : Nat
    g : GenContFract K
    inst✝ : Field K
    nth_partDen_ne_zero : ∀ {b : K}, Eq (g.partDens.get? n) (Option.some b) → Ne b 0
    ⊢ Eq (g.convs (HAdd.hAdd n 1)) ((g.squashGCF n).convs n)
  -/
  rcases Decidable.em (g.TerminatedAt n) with terminatedAt_n | not_terminatedAt_n
    /-
      case inl
      K : Type u_1
      n : Nat
      g : GenContFract K
      inst✝ : Field K
      nth_partDen_ne_zero : ∀ {b : K}, Eq (g.partDens.get? n) (Option.some b) → Ne b 0
      terminatedAt_n : g.TerminatedAt n
      ⊢ Eq (g.convs (HAdd.hAdd n 1)) ((g.squashGCF n).convs n)
    -/
  · have : squashGCF g n = g := squashGCF_eq_self_of_terminated terminatedAt_n
    /-
      case inl
      K : Type u_1
      n : Nat
      g : GenContFract K
      inst✝ : Field K
      nth_partDen_ne_zero : ∀ {b : K}, Eq (g.partDens.get? n) (Option.some b) → Ne b 0
      terminatedAt_n : g.TerminatedAt n
      this : Eq (g.squashGCF n) g
      ⊢ Eq (g.convs (HAdd.hAdd n 1)) ((g.squashGCF n).convs n)
    -/
    simp only [this, convs_stable_of_terminated n.le_succ terminatedAt_n]
    /-
      🎉 no goals
    -/
  · obtain ⟨⟨a, b⟩, s_nth_eq⟩ : ∃ gp_n, g.s.get? n = some gp_n :=
      Option.ne_none_iff_exists'.mp not_terminatedAt_n
    /-
      case inr.intro.mk
      K : Type u_1
      n : Nat
      g : GenContFract K
      inst✝ : Field K
      nth_partDen_ne_zero : ∀ {b : K}, Eq (g.partDens.get? n) (Option.some b) → Ne b 0
      not_terminatedAt_n : Not (g.TerminatedAt n)
      a b : K
      s_nth_eq : Eq (g.s.get? n) (Option.some { a := a, b := b })
      ⊢ Eq (g.convs (HAdd.hAdd n 1)) ((g.squashGCF n).convs n)
    -/
    have b_ne_zero : b ≠ 0 := nth_partDen_ne_zero (partDen_eq_s_b s_nth_eq)
    cases n with
    | zero =>
      suffices (b * g.h + a) / b = g.h + a / b by
        simpa [squashGCF, s_nth_eq, conv_eq_conts_a_div_conts_b,
          conts_recurrenceAux s_nth_eq zeroth_contAux_eq_one_zero first_contAux_eq_h_one]
      calc
        (b * g.h + a) / b = b * g.h / b + a / b := by ring
        -- requires `Field`, not `DivisionRing`
        _ = g.h + a / b := by rw [mul_div_cancel_left₀ _ b_ne_zero]
    | succ n' =>
      obtain ⟨⟨pa, pb⟩, s_n'th_eq⟩ : ∃ gp_n', g.s.get? n' = some gp_n' :=
        g.s.ge_stable n'.le_succ s_nth_eq
      -- Notations
      let g' := squashGCF g (n' + 1)
      set pred_conts := g.contsAux (n' + 1) with succ_n'th_contsAux_eq
      set ppred_conts := g.contsAux n' with n'th_contsAux_eq
      let pA := pred_conts.a
      let pB := pred_conts.b
      let ppA := ppred_conts.a
      let ppB := ppred_conts.b
      set pred_conts' := g'.contsAux (n' + 1) with succ_n'th_contsAux_eq'
      set ppred_conts' := g'.contsAux n' with n'th_contsAux_eq'
      let pA' := pred_conts'.a
      let pB' := pred_conts'.b
      let ppA' := ppred_conts'.a
      let ppB' := ppred_conts'.b
      -- first compute the convergent of the squashed gcf
      have : g'.convs (n' + 1) =
          ((pb + a / b) * pA' + pa * ppA') / ((pb + a / b) * pB' + pa * ppB') := by
        have : g'.s.get? n' = some ⟨pa, pb + a / b⟩ :=
          squashSeq_nth_of_not_terminated s_n'th_eq s_nth_eq
        rw [conv_eq_conts_a_div_conts_b,
          conts_recurrenceAux this n'th_contsAux_eq'.symm succ_n'th_contsAux_eq'.symm]
      rw [this]
      -- then compute the convergent of the original gcf by recursively unfolding the continuants
      -- computation twice
      have : g.convs (n' + 2) =
          (b * (pb * pA + pa * ppA) + a * pA) / (b * (pb * pB + pa * ppB) + a * pB) := by
        -- use the recurrence once
        have : g.contsAux (n' + 2) = ⟨pb * pA + pa * ppA, pb * pB + pa * ppB⟩ :=
          contsAux_recurrence s_n'th_eq n'th_contsAux_eq.symm succ_n'th_contsAux_eq.symm
        -- and a second time
        rw [conv_eq_conts_a_div_conts_b,
          conts_recurrenceAux s_nth_eq succ_n'th_contsAux_eq.symm this]
      rw [this]
      suffices
        ((pb + a / b) * pA + pa * ppA) / ((pb + a / b) * pB + pa * ppB) =
          (b * (pb * pA + pa * ppA) + a * pA) / (b * (pb * pB + pa * ppB) + a * pB) by
        obtain ⟨eq1, eq2, eq3, eq4⟩ : pA' = pA ∧ pB' = pB ∧ ppA' = ppA ∧ ppB' = ppB := by
          simp [*, g', pA, pB, ppA, ppB, pA', pB', ppA', ppB',
            (contsAux_eq_contsAux_squashGCF_of_le <| le_refl <| n' + 1).symm,
            (contsAux_eq_contsAux_squashGCF_of_le n'.le_succ).symm]
        symm
        simpa only [eq1, eq2, eq3, eq4, mul_div_cancel_right₀ _ b_ne_zero]
      field_simp
      congr 1 <;> ring


/-- Shows that the recurrence relation (`convs`) and direct evaluation (`convs'`) of the
generalized continued fraction coincide at position `n` if the sequence of fractions contains
strictly positive values only.
Requiring positivity of all values is just one possible condition to obtain this result.
For example, the dual - sequences with strictly negative values only - would also work.

In practice, one most commonly deals with regular continued fractions, which satisfy the
positivity criterion required here. The analogous result for them
(see `ContFract.convs_eq_convs`) hence follows directly from this theorem.
-/
theorem convs_eq_convs' [LinearOrderedField K]
    (s_pos : ∀ {gp : Pair K} {m : ℕ}, m < n → g.s.get? m = some gp → 0 < gp.a ∧ 0 < gp.b) :
    g.convs n = g.convs' n := by
  induction n generalizing g with
  | zero => simp
  | succ n IH =>
    let g' := squashGCF g n
    -- first replace the rhs with the squashed computation
    suffices g.convs (n + 1) = g'.convs' n by
      rwa [succ_nth_conv'_eq_squashGCF_nth_conv']
    rcases Decidable.em (TerminatedAt g n) with terminatedAt_n | not_terminatedAt_n
    · have g'_eq_g : g' = g := squashGCF_eq_self_of_terminated terminatedAt_n
      rw [convs_stable_of_terminated n.le_succ terminatedAt_n, g'_eq_g, IH _]
      intro _ _ m_lt_n s_mth_eq
      exact s_pos (Nat.lt.step m_lt_n) s_mth_eq
    · suffices g.convs (n + 1) = g'.convs n by
        -- invoke the IH for the squashed gcf
        rwa [← IH]
        intro gp' m m_lt_n s_mth_eq'
        -- case distinction on m + 1 = n or m + 1 < n
        rcases m_lt_n with n | succ_m_lt_n
        · -- the difficult case at the squashed position: we first obtain the values from
          -- the sequence
          obtain ⟨gp_succ_m, s_succ_mth_eq⟩ : ∃ gp_succ_m, g.s.get? (m + 1) = some gp_succ_m :=
            Option.ne_none_iff_exists'.mp not_terminatedAt_n
          obtain ⟨gp_m, mth_s_eq⟩ : ∃ gp_m, g.s.get? m = some gp_m :=
            g.s.ge_stable m.le_succ s_succ_mth_eq
          -- we then plug them into the recurrence
          suffices 0 < gp_m.a ∧ 0 < gp_m.b + gp_succ_m.a / gp_succ_m.b by
            have ot : g'.s.get? m = some ⟨gp_m.a, gp_m.b + gp_succ_m.a / gp_succ_m.b⟩ :=
              squashSeq_nth_of_not_terminated mth_s_eq s_succ_mth_eq
            have : gp' = ⟨gp_m.a, gp_m.b + gp_succ_m.a / gp_succ_m.b⟩ := by
              simp_all only [Option.some.injEq]
            rwa [this]
          have m_lt_n : m < m.succ := Nat.lt_succ_self m
          refine ⟨(s_pos (Nat.lt.step m_lt_n) mth_s_eq).left, ?_⟩
          refine add_pos (s_pos (Nat.lt.step m_lt_n) mth_s_eq).right ?_
          have : 0 < gp_succ_m.a ∧ 0 < gp_succ_m.b := s_pos (lt_add_one <| m + 1) s_succ_mth_eq
          exact div_pos this.left this.right
        · -- the easy case: before the squashed position, nothing changes
          refine s_pos (Nat.lt.step <| Nat.lt.step succ_m_lt_n) ?_
          exact Eq.trans (squashGCF_nth_of_lt succ_m_lt_n).symm s_mth_eq'
      -- now the result follows from the fact that the convergents coincide at the squashed position
      -- as established in `succ_nth_conv_eq_squashGCF_nth_conv`.
      have : ∀ ⦃b⦄, g.partDens.get? n = some b → b ≠ 0 := by
        intro b nth_partDen_eq
        obtain ⟨gp, s_nth_eq, ⟨refl⟩⟩ : ∃ gp, g.s.get? n = some gp ∧ gp.b = b :=
          exists_s_b_of_partDen nth_partDen_eq
        exact (ne_of_lt (s_pos (lt_add_one n) s_nth_eq).right).symm
      exact succ_nth_conv_eq_squashGCF_nth_conv @this


/-- Shows that the recurrence relation (`convs`) and direct evaluation (`convs'`) of a
(regular) continued fraction coincide. -/
nonrec theorem convs_eq_convs' [LinearOrderedField K] {c : ContFract K} :
    (↑c : GenContFract K).convs = (↑c : GenContFract K).convs' := by
  /-
    K : Type u_1
    inst✝ : LinearOrderedField K
    c : ContFract K
    ⊢ Eq (↑↑c).convs (↑↑c).convs'
  -/
  ext n
  /-
    case a
    K : Type u_1
    inst✝ : LinearOrderedField K
    c : ContFract K
    n : Nat
    ⊢ Eq ((↑↑c).convs.get n) (Stream'.get (↑↑c).convs' n)
  -/
  apply convs_eq_convs'
  /-
    case a.s_pos
    K : Type u_1
    inst✝ : LinearOrderedField K
    c : ContFract K
    n : Nat
    ⊢ ∀ {gp : GenContFract.Pair K} {m : Nat}, LT.lt m n → Eq ((↑↑c).s.get? m) (Opt …
  -/
  intro gp m _ s_nth_eq
  exact ⟨zero_lt_one.trans_le ((c : SimpContFract K).property m gp.a
    (partNum_eq_s_a s_nth_eq)).symm.le, c.property m gp.b <| partDen_eq_s_b s_nth_eq⟩


