theorem rfind' {f} (hf : Nat.Partrec f) :
    Nat.Partrec
      (Nat.unpaired fun a m =>
        (Nat.rfind fun n => (fun m => m = 0) <$> f (Nat.pair a (n + m))).map (· + m)) :=
  Partrec₂.unpaired'.2 <| by
    refine
      Partrec.map
        ((@Partrec₂.unpaired' fun a b : ℕ =>
              Nat.rfind fun n => (fun m => m = 0) <$> f (Nat.pair a (n + b))).1
          ?_)
        (Primrec.nat_add.comp Primrec.snd <| Primrec.snd.comp Primrec.fst).to_comp.to₂
    have : Nat.Partrec (fun a => Nat.rfind (fun n => (fun m => decide (m = 0)) <$>
      Nat.unpaired (fun a b => f (Nat.pair (Nat.unpair a).1 (b + (Nat.unpair a).2)))
        (Nat.pair a n))) :=
      rfind
        (Partrec₂.unpaired'.2
          ((Partrec.nat_iff.2 hf).comp
              (Primrec₂.pair.comp (Primrec.fst.comp <| Primrec.unpair.comp Primrec.fst)
                  (Primrec.nat_add.comp Primrec.snd
                    (Primrec.snd.comp <| Primrec.unpair.comp Primrec.fst))).to_comp))
    /-
      f : PFun Nat Nat
      hf : Nat.Partrec f
      this : Nat.Partrec fun a => Nat.rfind fun n => Functor.map (fun m => Decidable …
      ⊢ Nat.Partrec (Nat.unpaired fun a b => Nat.rfind fun n => Functor.map (fun m = …
    -/
    simpa
    /-
      🎉 no goals
    -/


/-- Code for partial recursive functions from ℕ to ℕ.
See `Nat.Partrec.Code.eval` for the interpretation of these constructors.
-/
inductive Code : Type
  | zero : Code
  | succ : Code
  | left : Code
  | right : Code
  | pair : Code → Code → Code
  | comp : Code → Code → Code
  | prec : Code → Code → Code
  | rfind' : Code → Code


compile_inductive% Code


instance instInhabited : Inhabited Code :=
  ⟨zero⟩


/-- Returns a code for the constant function outputting a particular natural. -/
protected def const : ℕ → Code
  | 0 => zero
  | n + 1 => comp succ (Code.const n)


theorem const_inj : ∀ {n₁ n₂}, Nat.Partrec.Code.const n₁ = Nat.Partrec.Code.const n₂ → n₁ = n₂
                  /-
                    x✝ : Eq (Nat.Partrec.Code.const 0) (Nat.Partrec.Code.const 0)
                    ⊢ Eq 0 0
                  -/
  | 0, 0, _ => by simp
                  /-
                    🎉 no goals
                  -/
  | n₁ + 1, n₂ + 1, h => by
    /-
      n₁ n₂ : Nat
      h : Eq (Nat.Partrec.Code.const (HAdd.hAdd n₁ 1)) (Nat.Partrec.Code.const (HAdd …
      ⊢ Eq (HAdd.hAdd n₁ 1) (HAdd.hAdd n₂ 1)
    -/
    dsimp [Nat.Partrec.Code.const] at h
    /-
      n₁ n₂ : Nat
      h : Eq (Nat.Partrec.Code.succ.comp (Nat.Partrec.Code.const n₁)) (Nat.Partrec.C …
      ⊢ Eq (HAdd.hAdd n₁ 1) (HAdd.hAdd n₂ 1)
    -/
    injection h with h₁ h₂
    /-
      n₁ n₂ : Nat
      h₁ : Eq Nat.Partrec.Code.succ Nat.Partrec.Code.succ
      h₂ : Eq (Nat.Partrec.Code.const n₁) (Nat.Partrec.Code.const n₂)
      ⊢ Eq (HAdd.hAdd n₁ 1) (HAdd.hAdd n₂ 1)
    -/
    simp only [const_inj h₂]
    /-
      🎉 no goals
    -/


/-- A code for the identity function. -/
protected def id : Code :=
  pair left right


/-- Given a code `c` taking a pair as input, returns a code using `n` as the first argument to `c`.
-/
def curry (c : Code) (n : ℕ) : Code :=
  comp c (pair (Code.const n) Code.id)


/-- An encoding of a `Nat.Partrec.Code` as a ℕ. -/
def encodeCode : Code → ℕ
  | zero => 0
  | succ => 1
  | left => 2
  | right => 3
  | pair cf cg => 2 * (2 * Nat.pair (encodeCode cf) (encodeCode cg)) + 4
  | comp cf cg => 2 * (2 * Nat.pair (encodeCode cf) (encodeCode cg) + 1) + 4
  | prec cf cg => (2 * (2 * Nat.pair (encodeCode cf) (encodeCode cg)) + 1) + 4
  | rfind' cf => (2 * (2 * encodeCode cf + 1) + 1) + 4


/--
A decoder for `Nat.Partrec.Code.encodeCode`, taking any ℕ to the `Nat.Partrec.Code` it represents.
-/
def ofNatCode : ℕ → Code
  | 0 => zero
  | 1 => succ
  | 2 => left
  | 3 => right
  | n + 4 =>
    let m := n.div2.div2
    have hm : m < n + 4 := by
      /-
        n : Nat
        m : Nat := n.div2.div2
        ⊢ LT.lt m (HAdd.hAdd n 4)
      -/
      simp only [m, div2_val]
      exact
        lt_of_le_of_lt (le_trans (Nat.div_le_self _ _) (Nat.div_le_self _ _))
          (Nat.succ_le_succ (Nat.le_add_right _ _))
    have _m1 : m.unpair.1 < n + 4 := lt_of_le_of_lt m.unpair_left_le hm
    have _m2 : m.unpair.2 < n + 4 := lt_of_le_of_lt m.unpair_right_le hm
    match n.bodd, n.div2.bodd with
    | false, false => pair (ofNatCode m.unpair.1) (ofNatCode m.unpair.2)
    | false, true  => comp (ofNatCode m.unpair.1) (ofNatCode m.unpair.2)
    | true , false => prec (ofNatCode m.unpair.1) (ofNatCode m.unpair.2)
    | true , true  => rfind' (ofNatCode m)


/-- Proof that `Nat.Partrec.Code.ofNatCode` is the inverse of `Nat.Partrec.Code.encodeCode`-/
private theorem encode_ofNatCode : ∀ n, encodeCode (ofNatCode n) = n
            /-
              ⊢ Eq (Nat.Partrec.Code.ofNatCode 0).encodeCode 0
            -/
  | 0 => by simp [ofNatCode, encodeCode]
            /-
              🎉 no goals
            -/
            /-
              ⊢ Eq (Nat.Partrec.Code.ofNatCode 1).encodeCode 1
            -/
  | 1 => by simp [ofNatCode, encodeCode]
            /-
              🎉 no goals
            -/
            /-
              ⊢ Eq (Nat.Partrec.Code.ofNatCode 2).encodeCode 2
            -/
  | 2 => by simp [ofNatCode, encodeCode]
            /-
              🎉 no goals
            -/
            /-
              ⊢ Eq (Nat.Partrec.Code.ofNatCode 3).encodeCode 3
            -/
  | 3 => by simp [ofNatCode, encodeCode]
            /-
              🎉 no goals
            -/
  | n + 4 => by
    /-
      n : Nat
      ⊢ Eq (Nat.Partrec.Code.ofNatCode (HAdd.hAdd n 4)).encodeCode (HAdd.hAdd n 4)
    -/
    let m := n.div2.div2
    have hm : m < n + 4 := by
      simp only [m, div2_val]
      exact
        lt_of_le_of_lt (le_trans (Nat.div_le_self _ _) (Nat.div_le_self _ _))
          (Nat.succ_le_succ (Nat.le_add_right _ _))
    /-
      n : Nat
      m : Nat := n.div2.div2
      hm : LT.lt m (HAdd.hAdd n 4)
      ⊢ Eq (Nat.Partrec.Code.ofNatCode (HAdd.hAdd n 4)).encodeCode (HAdd.hAdd n 4)
    -/
    have _m1 : m.unpair.1 < n + 4 := lt_of_le_of_lt m.unpair_left_le hm
    /-
      n : Nat
      m : Nat := n.div2.div2
      hm : LT.lt m (HAdd.hAdd n 4)
      _m1 : LT.lt (Nat.unpair m).1 (HAdd.hAdd n 4)
      ⊢ Eq (Nat.Partrec.Code.ofNatCode (HAdd.hAdd n 4)).encodeCode (HAdd.hAdd n 4)
    -/
    have _m2 : m.unpair.2 < n + 4 := lt_of_le_of_lt m.unpair_right_le hm
    /-
      n : Nat
      m : Nat := n.div2.div2
      hm : LT.lt m (HAdd.hAdd n 4)
      _m1 : LT.lt (Nat.unpair m).1 (HAdd.hAdd n 4)
      _m2 : LT.lt (Nat.unpair m).2 (HAdd.hAdd n 4)
      ⊢ Eq (Nat.Partrec.Code.ofNatCode (HAdd.hAdd n 4)).encodeCode (HAdd.hAdd n 4)
    -/
    have IH := encode_ofNatCode m
    /-
      n : Nat
      m : Nat := n.div2.div2
      hm : LT.lt m (HAdd.hAdd n 4)
      _m1 : LT.lt (Nat.unpair m).1 (HAdd.hAdd n 4)
      _m2 : LT.lt (Nat.unpair m).2 (HAdd.hAdd n 4)
      IH : Eq (Nat.Partrec.Code.ofNatCode m).encodeCode m
      ⊢ Eq (Nat.Partrec.Code.ofNatCode (HAdd.hAdd n 4)).encodeCode (HAdd.hAdd n 4)
    -/
    have IH1 := encode_ofNatCode m.unpair.1
    /-
      n : Nat
      m : Nat := n.div2.div2
      hm : LT.lt m (HAdd.hAdd n 4)
      _m1 : LT.lt (Nat.unpair m).1 (HAdd.hAdd n 4)
      _m2 : LT.lt (Nat.unpair m).2 (HAdd.hAdd n 4)
      IH : Eq (Nat.Partrec.Code.ofNatCode m).encodeCode m
      IH1 : Eq (Nat.Partrec.Code.ofNatCode (Nat.unpair m).1).encodeCode (Nat.unpair  …
      ⊢ Eq (Nat.Partrec.Code.ofNatCode (HAdd.hAdd n 4)).encodeCode (HAdd.hAdd n 4)
    -/
    have IH2 := encode_ofNatCode m.unpair.2
    /-
      n : Nat
      m : Nat := n.div2.div2
      hm : LT.lt m (HAdd.hAdd n 4)
      _m1 : LT.lt (Nat.unpair m).1 (HAdd.hAdd n 4)
      _m2 : LT.lt (Nat.unpair m).2 (HAdd.hAdd n 4)
      IH : Eq (Nat.Partrec.Code.ofNatCode m).encodeCode m
      IH1 : Eq (Nat.Partrec.Code.ofNatCode (Nat.unpair m).1).encodeCode (Nat.unpair  …
      IH2 : Eq (Nat.Partrec.Code.ofNatCode (Nat.unpair m).2).encodeCode (Nat.unpair  …
      ⊢ Eq (Nat.Partrec.Code.ofNatCode (HAdd.hAdd n 4)).encodeCode (HAdd.hAdd n 4)
    -/
    conv_rhs => rw [← Nat.bit_decomp n, ← Nat.bit_decomp n.div2]
    /-
      n : Nat
      m : Nat := n.div2.div2
      hm : LT.lt m (HAdd.hAdd n 4)
      _m1 : LT.lt (Nat.unpair m).1 (HAdd.hAdd n 4)
      _m2 : LT.lt (Nat.unpair m).2 (HAdd.hAdd n 4)
      IH : Eq (Nat.Partrec.Code.ofNatCode m).encodeCode m
      IH1 : Eq (Nat.Partrec.Code.ofNatCode (Nat.unpair m).1).encodeCode (Nat.unpair  …
      IH2 : Eq (Nat.Partrec.Code.ofNatCode (Nat.unpair m).2).encodeCode (Nat.unpair  …
      ⊢ Eq (Nat.Partrec.Code.ofNatCode (HAdd.hAdd n 4)).encodeCode (HAdd.hAdd (Nat.b …
    -/
    simp only [ofNatCode.eq_5]
    /-
      n : Nat
      m : Nat := n.div2.div2
      hm : LT.lt m (HAdd.hAdd n 4)
      _m1 : LT.lt (Nat.unpair m).1 (HAdd.hAdd n 4)
      _m2 : LT.lt (Nat.unpair m).2 (HAdd.hAdd n 4)
      IH : Eq (Nat.Partrec.Code.ofNatCode m).encodeCode m
      IH1 : Eq (Nat.Partrec.Code.ofNatCode (Nat.unpair m).1).encodeCode (Nat.unpair  …
      IH2 : Eq (Nat.Partrec.Code.ofNatCode (Nat.unpair m).2).encodeCode (Nat.unpair  …
      ⊢ Eq (Nat.Partrec.Code.ofNatCode.match_1 (fun x x => Nat.Partrec.Code) n.bodd  …
    -/
    cases n.bodd <;> cases n.div2.bodd <;>
      /-
        case false.false
        n : Nat
        m : Nat := n.div2.div2
        hm : LT.lt m (HAdd.hAdd n 4)
        _m1 : LT.lt (Nat.unpair m).1 (HAdd.hAdd n 4)
        _m2 : LT.lt (Nat.unpair m).2 (HAdd.hAdd n 4)
        IH : Eq (Nat.Partrec.Code.ofNatCode m).encodeCode m
        IH1 : Eq (Nat.Partrec.Code.ofNatCode (Nat.unpair m).1).encodeCode (Nat.unpair  …
        IH2 : Eq (Nat.Partrec.Code.ofNatCode (Nat.unpair m).2).encodeCode (Nat.unpair  …
        ⊢ Eq (Nat.Partrec.Code.ofNatCode.match_1 (fun x x => Nat.Partrec.Code) Bool.fa …
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
      simp [m, encodeCode, ofNatCode, IH, IH1, IH2, Nat.bit_val]
      /-
        🎉 no goals
      -/


instance instDenumerable : Denumerable Code :=
  mk'
    ⟨encodeCode, ofNatCode, fun c => by
        /-
          c : Nat.Partrec.Code
          ⊢ Eq (Nat.Partrec.Code.ofNatCode c.encodeCode) c
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
                          🎉 no goals
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
        induction c <;> simp [encodeCode, ofNatCode, Nat.div2_val, *],
                        /-
                          🎉 no goals
                        -/
      encode_ofNatCode⟩


theorem encodeCode_eq : encode = encodeCode :=
  rfl


theorem ofNatCode_eq : ofNat Code = ofNatCode :=
  rfl


theorem encode_lt_pair (cf cg) :
    encode cf < encode (pair cf cg) ∧ encode cg < encode (pair cf cg) := by
  /-
    cf cg : Nat.Partrec.Code
    ⊢ And (LT.lt (Encodable.encode cf) (Encodable.encode (cf.pair cg))) (LT.lt (En …
  -/
  simp only [encodeCode_eq, encodeCode]
  /-
    cf cg : Nat.Partrec.Code
    ⊢ And (LT.lt cf.encodeCode (HAdd.hAdd (HMul.hMul 2 (HMul.hMul 2 (Nat.pair cf.e …
  -/
  have := Nat.mul_le_mul_right (Nat.pair cf.encodeCode cg.encodeCode) (by decide : 1 ≤ 2 * 2)
  /-
    cf cg : Nat.Partrec.Code
    this : LE.le (HMul.hMul 1 (Nat.pair cf.encodeCode cg.encodeCode)) (HMul.hMul ( …
    ⊢ And (LT.lt cf.encodeCode (HAdd.hAdd (HMul.hMul 2 (HMul.hMul 2 (Nat.pair cf.e …
  -/
  rw [one_mul, mul_assoc] at this
  /-
    cf cg : Nat.Partrec.Code
    this : LE.le (Nat.pair cf.encodeCode cg.encodeCode) (HMul.hMul 2 (HMul.hMul 2  …
    ⊢ And (LT.lt cf.encodeCode (HAdd.hAdd (HMul.hMul 2 (HMul.hMul 2 (Nat.pair cf.e …
  -/
  have := lt_of_le_of_lt this (lt_add_of_pos_right _ (by decide : 0 < 4))
  /-
    cf cg : Nat.Partrec.Code
    this✝ : LE.le (Nat.pair cf.encodeCode cg.encodeCode) (HMul.hMul 2 (HMul.hMul 2 …
    this : LT.lt (Nat.pair cf.encodeCode cg.encodeCode) (HAdd.hAdd (HMul.hMul 2 (H …
    ⊢ And (LT.lt cf.encodeCode (HAdd.hAdd (HMul.hMul 2 (HMul.hMul 2 (Nat.pair cf.e …
  -/
  exact ⟨lt_of_le_of_lt (Nat.left_le_pair _ _) this, lt_of_le_of_lt (Nat.right_le_pair _ _) this⟩
  /-
    🎉 no goals
  -/


theorem encode_lt_comp (cf cg) :
    encode cf < encode (comp cf cg) ∧ encode cg < encode (comp cf cg) := by
  /-
    cf cg : Nat.Partrec.Code
    ⊢ And (LT.lt (Encodable.encode cf) (Encodable.encode (cf.comp cg))) (LT.lt (En …
  -/
  have : encode (pair cf cg) < encode (comp cf cg) := by simp [encodeCode_eq, encodeCode]
  /-
    cf cg : Nat.Partrec.Code
    this : LT.lt (Encodable.encode (cf.pair cg)) (Encodable.encode (cf.comp cg))
    ⊢ And (LT.lt (Encodable.encode cf) (Encodable.encode (cf.comp cg))) (LT.lt (En …
  -/
  exact (encode_lt_pair cf cg).imp (fun h => lt_trans h this) fun h => lt_trans h this
  /-
    🎉 no goals
  -/


theorem encode_lt_prec (cf cg) :
    encode cf < encode (prec cf cg) ∧ encode cg < encode (prec cf cg) := by
  /-
    cf cg : Nat.Partrec.Code
    ⊢ And (LT.lt (Encodable.encode cf) (Encodable.encode (cf.prec cg))) (LT.lt (En …
  -/
  have : encode (pair cf cg) < encode (prec cf cg) := by simp [encodeCode_eq, encodeCode]
  /-
    cf cg : Nat.Partrec.Code
    this : LT.lt (Encodable.encode (cf.pair cg)) (Encodable.encode (cf.prec cg))
    ⊢ And (LT.lt (Encodable.encode cf) (Encodable.encode (cf.prec cg))) (LT.lt (En …
  -/
  exact (encode_lt_pair cf cg).imp (fun h => lt_trans h this) fun h => lt_trans h this
  /-
    🎉 no goals
  -/


theorem encode_lt_rfind' (cf) : encode cf < encode (rfind' cf) := by
  /-
    cf : Nat.Partrec.Code
    ⊢ LT.lt (Encodable.encode cf) (Encodable.encode cf.rfind')
  -/
  simp only [encodeCode_eq, encodeCode]
  /-
    cf : Nat.Partrec.Code
    ⊢ LT.lt cf.encodeCode (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd (HMul.hMul …
  -/
  have := Nat.mul_le_mul_right cf.encodeCode (by decide : 1 ≤ 2 * 2)
  /-
    cf : Nat.Partrec.Code
    this : LE.le (HMul.hMul 1 cf.encodeCode) (HMul.hMul (HMul.hMul 2 2) cf.encodeC …
    ⊢ LT.lt cf.encodeCode (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd (HMul.hMul …
  -/
  rw [one_mul, mul_assoc] at this
  /-
    cf : Nat.Partrec.Code
    this : LE.le cf.encodeCode (HMul.hMul 2 (HMul.hMul 2 cf.encodeCode))
    ⊢ LT.lt cf.encodeCode (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd (HMul.hMul …
  -/
  refine lt_of_le_of_lt (le_trans this ?_) (lt_add_of_pos_right _ (by decide : 0 < 4))
  exact le_of_lt (Nat.lt_succ_of_le <| Nat.mul_le_mul_left _ <| le_of_lt <|
    Nat.lt_succ_of_le <| Nat.mul_le_mul_left _ <| le_rfl)


theorem pair_prim : Primrec₂ pair :=
  Primrec₂.ofNat_iff.2 <|
    Primrec₂.encode_iff.1 <|
      nat_add.comp
        (nat_double.comp <|
          nat_double.comp <|
            Primrec₂.natPair.comp (encode_iff.2 <| (Primrec.ofNat Code).comp fst)
              (encode_iff.2 <| (Primrec.ofNat Code).comp snd))
        (Primrec₂.const 4)


theorem comp_prim : Primrec₂ comp :=
  Primrec₂.ofNat_iff.2 <|
    Primrec₂.encode_iff.1 <|
      nat_add.comp
        (nat_double.comp <|
          nat_double_succ.comp <|
            Primrec₂.natPair.comp (encode_iff.2 <| (Primrec.ofNat Code).comp fst)
              (encode_iff.2 <| (Primrec.ofNat Code).comp snd))
        (Primrec₂.const 4)


theorem prec_prim : Primrec₂ prec :=
  Primrec₂.ofNat_iff.2 <|
    Primrec₂.encode_iff.1 <|
      nat_add.comp
        (nat_double_succ.comp <|
          nat_double.comp <|
            Primrec₂.natPair.comp (encode_iff.2 <| (Primrec.ofNat Code).comp fst)
              (encode_iff.2 <| (Primrec.ofNat Code).comp snd))
        (Primrec₂.const 4)


theorem rfind_prim : Primrec rfind' :=
  ofNat_iff.2 <|
    encode_iff.1 <|
      nat_add.comp
        (nat_double_succ.comp <| nat_double_succ.comp <|
          encode_iff.2 <| Primrec.ofNat Code)
        (const 4)


theorem rec_prim' {α σ} [Primcodable α] [Primcodable σ] {c : α → Code} (hc : Primrec c) {z : α → σ}
    (hz : Primrec z) {s : α → σ} (hs : Primrec s) {l : α → σ} (hl : Primrec l) {r : α → σ}
    (hr : Primrec r) {pr : α → Code × Code × σ × σ → σ} (hpr : Primrec₂ pr)
    {co : α → Code × Code × σ × σ → σ} (hco : Primrec₂ co) {pc : α → Code × Code × σ × σ → σ}
    (hpc : Primrec₂ pc) {rf : α → Code × σ → σ} (hrf : Primrec₂ rf) :
    let PR (a) cf cg hf hg := pr a (cf, cg, hf, hg)
    let CO (a) cf cg hf hg := co a (cf, cg, hf, hg)
    let PC (a) cf cg hf hg := pc a (cf, cg, hf, hg)
    let RF (a) cf hf := rf a (cf, hf)
    let F (a : α) (c : Code) : σ :=
      Nat.Partrec.Code.recOn c (z a) (s a) (l a) (r a) (PR a) (CO a) (PC a) (RF a)
    Primrec (fun a => F a (c a) : α → σ) := by
  /-
    α : Type u_1
    σ : Type u_2
    inst✝¹ : Primcodable α
    inst✝ : Primcodable σ
    c : α → Nat.Partrec.Code
    hc : Primrec c
    z : α → σ
    hz : Primrec z
    s : α → σ
    hs : Primrec s
    l : α → σ
    hl : Primrec l
    r : α → σ
    hr : Primrec r
    pr : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpr : Primrec₂ pr
    co : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hco : Primrec₂ co
    pc : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpc : Primrec₂ pc
    rf : α → Prod Nat.Partrec.Code σ → σ
    hrf : Primrec₂ rf
    ⊢ let PR := fun a cf cg hf hg => pr a { fst := cf, snd := { fst := cg, snd :=  …
      let CO := fun a cf cg hf hg => co a { fst := cf, snd := { fst := cg, snd :=  …
      let PC := fun a cf cg hf hg => pc a { fst := cf, snd := { fst := cg, snd :=  …
      let RF := fun a cf hf => rf a { fst := cf, snd := hf };
      let F := fun a c => Nat.Partrec.Code.recOn c (z a) (s a) (l a) (r a) (PR a)  …
      Primrec fun a => F a (c a)
  -/
  intros _ _ _ _ F
  let G₁ : (α × List σ) × ℕ × ℕ → Option σ := fun p =>
    letI a := p.1.1; letI IH := p.1.2; letI n := p.2.1; letI m := p.2.2
    (IH.get? m).bind fun s =>
    (IH.get? m.unpair.1).bind fun s₁ =>
    (IH.get? m.unpair.2).map fun s₂ =>
    cond n.bodd
      (cond n.div2.bodd (rf a (ofNat Code m, s))
        (pc a (ofNat Code m.unpair.1, ofNat Code m.unpair.2, s₁, s₂)))
      (cond n.div2.bodd (co a (ofNat Code m.unpair.1, ofNat Code m.unpair.2, s₁, s₂))
        (pr a (ofNat Code m.unpair.1, ofNat Code m.unpair.2, s₁, s₂)))
  have : Primrec G₁ :=
    option_bind (list_get?.comp (snd.comp fst) (snd.comp snd)) <| .mk <|
    option_bind ((list_get?.comp (snd.comp fst)
      (fst.comp <| Primrec.unpair.comp (snd.comp snd))).comp fst) <| .mk <|
    option_map ((list_get?.comp (snd.comp fst)
      (snd.comp <| Primrec.unpair.comp (snd.comp snd))).comp <| fst.comp fst) <| .mk <|
    have a := fst.comp (fst.comp <| fst.comp <| fst.comp fst)
    have n := fst.comp (snd.comp <| fst.comp <| fst.comp fst)
    have m := snd.comp (snd.comp <| fst.comp <| fst.comp fst)
    have m₁ := fst.comp (Primrec.unpair.comp m)
    have m₂ := snd.comp (Primrec.unpair.comp m)
    have s := snd.comp (fst.comp fst)
    have s₁ := snd.comp fst
    have s₂ := snd
    (nat_bodd.comp n).cond
      ((nat_bodd.comp <| nat_div2.comp n).cond
        (hrf.comp a (((Primrec.ofNat Code).comp m).pair s))
        (hpc.comp a (((Primrec.ofNat Code).comp m₁).pair <|
          ((Primrec.ofNat Code).comp m₂).pair <| s₁.pair s₂)))
      (Primrec.cond (nat_bodd.comp <| nat_div2.comp n)
        (hco.comp a (((Primrec.ofNat Code).comp m₁).pair <|
          ((Primrec.ofNat Code).comp m₂).pair <| s₁.pair s₂))
        (hpr.comp a (((Primrec.ofNat Code).comp m₁).pair <|
          ((Primrec.ofNat Code).comp m₂).pair <| s₁.pair s₂)))
  let G : α → List σ → Option σ := fun a IH =>
    IH.length.casesOn (some (z a)) fun n =>
    n.casesOn (some (s a)) fun n =>
    n.casesOn (some (l a)) fun n =>
    n.casesOn (some (r a)) fun n =>
    G₁ ((a, IH), n, n.div2.div2)
  have : Primrec₂ G := .mk <|
    nat_casesOn (list_length.comp snd) (option_some_iff.2 (hz.comp fst)) <| .mk <|
    nat_casesOn snd (option_some_iff.2 (hs.comp (fst.comp fst))) <| .mk <|
    nat_casesOn snd (option_some_iff.2 (hl.comp (fst.comp <| fst.comp fst))) <| .mk <|
    nat_casesOn snd (option_some_iff.2 (hr.comp (fst.comp <| fst.comp <| fst.comp fst))) <| .mk <|
    this.comp <|
      ((fst.pair snd).comp <| fst.comp <| fst.comp <| fst.comp <| fst).pair <|
      snd.pair <| nat_div2.comp <| nat_div2.comp snd
  refine (nat_strong_rec (fun a n => F a (ofNat Code n)) this.to₂ fun a n => ?_)
    |>.comp .id (encode_iff.2 hc) |>.of_eq fun a => by simp
  /-
    α : Type u_1
    σ : Type u_2
    inst✝¹ : Primcodable α
    inst✝ : Primcodable σ
    c : α → Nat.Partrec.Code
    hc : Primrec c
    z : α → σ
    hz : Primrec z
    s : α → σ
    hs : Primrec s
    l : α → σ
    hl : Primrec l
    r : α → σ
    hr : Primrec r
    pr : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpr : Primrec₂ pr
    co : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hco : Primrec₂ co
    pc : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpc : Primrec₂ pc
    rf : α → Prod Nat.Partrec.Code σ → σ
    hrf : Primrec₂ rf
    PR✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    CO✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    PC✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    RF✝ : α → Nat.Partrec.Code → σ → σ := fun a cf hf => rf a { fst := cf, snd :=  …
    F : α → Nat.Partrec.Code → σ := fun a c => Nat.Partrec.Code.recOn c (z a) (s a …
    G₁ : Prod (Prod α (List σ)) (Prod Nat Nat) → Option σ := fun p => (p.1.2.get?  …
    this✝ : Primrec G₁
    G : α → List σ → Option σ := fun a IH => Nat.casesOn IH.length (Option.some (z …
    this : Primrec₂ G
    a : α
    n : Nat
    ⊢ Eq (G { fst := a, snd := List.map ((fun a n => F a (Denumerable.ofNat Nat.Pa …
  -/
  iterate 4 cases' n with n; · simp [ofNatCode_eq, ofNatCode]; rfl
  /-
    case succ.succ.succ.succ
    α : Type u_1
    σ : Type u_2
    inst✝¹ : Primcodable α
    inst✝ : Primcodable σ
    c : α → Nat.Partrec.Code
    hc : Primrec c
    z : α → σ
    hz : Primrec z
    s : α → σ
    hs : Primrec s
    l : α → σ
    hl : Primrec l
    r : α → σ
    hr : Primrec r
    pr : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpr : Primrec₂ pr
    co : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hco : Primrec₂ co
    pc : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpc : Primrec₂ pc
    rf : α → Prod Nat.Partrec.Code σ → σ
    hrf : Primrec₂ rf
    PR✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    CO✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    PC✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    RF✝ : α → Nat.Partrec.Code → σ → σ := fun a cf hf => rf a { fst := cf, snd :=  …
    F : α → Nat.Partrec.Code → σ := fun a c => Nat.Partrec.Code.recOn c (z a) (s a …
    G₁ : Prod (Prod α (List σ)) (Prod Nat Nat) → Option σ := fun p => (p.1.2.get?  …
    this✝ : Primrec G₁
    G : α → List σ → Option σ := fun a IH => Nat.casesOn IH.length (Option.some (z …
    this : Primrec₂ G
    a : α
    n : Nat
    ⊢ Eq (G { fst := a, snd := List.map ((fun a n => F a (Denumerable.ofNat Nat.Pa …
  -/
  simp only [G]; rw [List.length_map, List.length_range]
  /-
    case succ.succ.succ.succ
    α : Type u_1
    σ : Type u_2
    inst✝¹ : Primcodable α
    inst✝ : Primcodable σ
    c : α → Nat.Partrec.Code
    hc : Primrec c
    z : α → σ
    hz : Primrec z
    s : α → σ
    hs : Primrec s
    l : α → σ
    hl : Primrec l
    r : α → σ
    hr : Primrec r
    pr : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpr : Primrec₂ pr
    co : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hco : Primrec₂ co
    pc : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpc : Primrec₂ pc
    rf : α → Prod Nat.Partrec.Code σ → σ
    hrf : Primrec₂ rf
    PR✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    CO✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    PC✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    RF✝ : α → Nat.Partrec.Code → σ → σ := fun a cf hf => rf a { fst := cf, snd :=  …
    F : α → Nat.Partrec.Code → σ := fun a c => Nat.Partrec.Code.recOn c (z a) (s a …
    G₁ : Prod (Prod α (List σ)) (Prod Nat Nat) → Option σ := fun p => (p.1.2.get?  …
    this✝ : Primrec G₁
    G : α → List σ → Option σ := fun a IH => Nat.casesOn IH.length (Option.some (z …
    this : Primrec₂ G
    a : α
    n : Nat
    ⊢ Eq (Nat.rec (Option.some (z a)) (fun n_1 n_ih => Nat.rec (Option.some (s a)) …
  -/
  let m := n.div2.div2
  show G₁ ((a, (List.range (n + 4)).map fun n => F a (ofNat Code n)), n, m)
    = some (F a (ofNat Code (n + 4)))
  have hm : m < n + 4 := by
    simp only [m, div2_val]
    exact lt_of_le_of_lt
      (le_trans (Nat.div_le_self ..) (Nat.div_le_self ..))
      (Nat.succ_le_succ (Nat.le_add_right ..))
  /-
    case succ.succ.succ.succ
    α : Type u_1
    σ : Type u_2
    inst✝¹ : Primcodable α
    inst✝ : Primcodable σ
    c : α → Nat.Partrec.Code
    hc : Primrec c
    z : α → σ
    hz : Primrec z
    s : α → σ
    hs : Primrec s
    l : α → σ
    hl : Primrec l
    r : α → σ
    hr : Primrec r
    pr : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpr : Primrec₂ pr
    co : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hco : Primrec₂ co
    pc : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpc : Primrec₂ pc
    rf : α → Prod Nat.Partrec.Code σ → σ
    hrf : Primrec₂ rf
    PR✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    CO✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    PC✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    RF✝ : α → Nat.Partrec.Code → σ → σ := fun a cf hf => rf a { fst := cf, snd :=  …
    F : α → Nat.Partrec.Code → σ := fun a c => Nat.Partrec.Code.recOn c (z a) (s a …
    G₁ : Prod (Prod α (List σ)) (Prod Nat Nat) → Option σ := fun p => (p.1.2.get?  …
    this✝ : Primrec G₁
    G : α → List σ → Option σ := fun a IH => Nat.casesOn IH.length (Option.some (z …
    this : Primrec₂ G
    a : α
    n : Nat
    m : Nat := n.div2.div2
    hm : LT.lt m (HAdd.hAdd n 4)
    ⊢ Eq (G₁ { fst := { fst := a, snd := List.map (fun n => F a (Denumerable.ofNat …
  -/
  have m1 : m.unpair.1 < n + 4 := lt_of_le_of_lt m.unpair_left_le hm
  /-
    case succ.succ.succ.succ
    α : Type u_1
    σ : Type u_2
    inst✝¹ : Primcodable α
    inst✝ : Primcodable σ
    c : α → Nat.Partrec.Code
    hc : Primrec c
    z : α → σ
    hz : Primrec z
    s : α → σ
    hs : Primrec s
    l : α → σ
    hl : Primrec l
    r : α → σ
    hr : Primrec r
    pr : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpr : Primrec₂ pr
    co : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hco : Primrec₂ co
    pc : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpc : Primrec₂ pc
    rf : α → Prod Nat.Partrec.Code σ → σ
    hrf : Primrec₂ rf
    PR✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    CO✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    PC✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    RF✝ : α → Nat.Partrec.Code → σ → σ := fun a cf hf => rf a { fst := cf, snd :=  …
    F : α → Nat.Partrec.Code → σ := fun a c => Nat.Partrec.Code.recOn c (z a) (s a …
    G₁ : Prod (Prod α (List σ)) (Prod Nat Nat) → Option σ := fun p => (p.1.2.get?  …
    this✝ : Primrec G₁
    G : α → List σ → Option σ := fun a IH => Nat.casesOn IH.length (Option.some (z …
    this : Primrec₂ G
    a : α
    n : Nat
    m : Nat := n.div2.div2
    hm : LT.lt m (HAdd.hAdd n 4)
    m1 : LT.lt (Nat.unpair m).1 (HAdd.hAdd n 4)
    ⊢ Eq (G₁ { fst := { fst := a, snd := List.map (fun n => F a (Denumerable.ofNat …
  -/
  have m2 : m.unpair.2 < n + 4 := lt_of_le_of_lt m.unpair_right_le hm
  /-
    case succ.succ.succ.succ
    α : Type u_1
    σ : Type u_2
    inst✝¹ : Primcodable α
    inst✝ : Primcodable σ
    c : α → Nat.Partrec.Code
    hc : Primrec c
    z : α → σ
    hz : Primrec z
    s : α → σ
    hs : Primrec s
    l : α → σ
    hl : Primrec l
    r : α → σ
    hr : Primrec r
    pr : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpr : Primrec₂ pr
    co : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hco : Primrec₂ co
    pc : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpc : Primrec₂ pc
    rf : α → Prod Nat.Partrec.Code σ → σ
    hrf : Primrec₂ rf
    PR✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    CO✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    PC✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    RF✝ : α → Nat.Partrec.Code → σ → σ := fun a cf hf => rf a { fst := cf, snd :=  …
    F : α → Nat.Partrec.Code → σ := fun a c => Nat.Partrec.Code.recOn c (z a) (s a …
    G₁ : Prod (Prod α (List σ)) (Prod Nat Nat) → Option σ := fun p => (p.1.2.get?  …
    this✝ : Primrec G₁
    G : α → List σ → Option σ := fun a IH => Nat.casesOn IH.length (Option.some (z …
    this : Primrec₂ G
    a : α
    n : Nat
    m : Nat := n.div2.div2
    hm : LT.lt m (HAdd.hAdd n 4)
    m1 : LT.lt (Nat.unpair m).1 (HAdd.hAdd n 4)
    m2 : LT.lt (Nat.unpair m).2 (HAdd.hAdd n 4)
    ⊢ Eq (G₁ { fst := { fst := a, snd := List.map (fun n => F a (Denumerable.ofNat …
  -/
  simp [G₁, m, List.getElem?_map, List.getElem?_range, hm, m1, m2]
  /-
    case succ.succ.succ.succ
    α : Type u_1
    σ : Type u_2
    inst✝¹ : Primcodable α
    inst✝ : Primcodable σ
    c : α → Nat.Partrec.Code
    hc : Primrec c
    z : α → σ
    hz : Primrec z
    s : α → σ
    hs : Primrec s
    l : α → σ
    hl : Primrec l
    r : α → σ
    hr : Primrec r
    pr : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpr : Primrec₂ pr
    co : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hco : Primrec₂ co
    pc : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpc : Primrec₂ pc
    rf : α → Prod Nat.Partrec.Code σ → σ
    hrf : Primrec₂ rf
    PR✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    CO✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    PC✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    RF✝ : α → Nat.Partrec.Code → σ → σ := fun a cf hf => rf a { fst := cf, snd :=  …
    F : α → Nat.Partrec.Code → σ := fun a c => Nat.Partrec.Code.recOn c (z a) (s a …
    G₁ : Prod (Prod α (List σ)) (Prod Nat Nat) → Option σ := fun p => (p.1.2.get?  …
    this✝ : Primrec G₁
    G : α → List σ → Option σ := fun a IH => Nat.casesOn IH.length (Option.some (z …
    this : Primrec₂ G
    a : α
    n : Nat
    m : Nat := n.div2.div2
    hm : LT.lt m (HAdd.hAdd n 4)
    m1 : LT.lt (Nat.unpair m).1 (HAdd.hAdd n 4)
    m2 : LT.lt (Nat.unpair m).2 (HAdd.hAdd n 4)
    ⊢ Eq (cond n.bodd (cond n.div2.bodd (rf a { fst := Denumerable.ofNat Nat.Partr …
  -/
  rw [show ofNat Code (n + 4) = ofNatCode (n + 4) from rfl]
  /-
    case succ.succ.succ.succ
    α : Type u_1
    σ : Type u_2
    inst✝¹ : Primcodable α
    inst✝ : Primcodable σ
    c : α → Nat.Partrec.Code
    hc : Primrec c
    z : α → σ
    hz : Primrec z
    s : α → σ
    hs : Primrec s
    l : α → σ
    hl : Primrec l
    r : α → σ
    hr : Primrec r
    pr : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpr : Primrec₂ pr
    co : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hco : Primrec₂ co
    pc : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpc : Primrec₂ pc
    rf : α → Prod Nat.Partrec.Code σ → σ
    hrf : Primrec₂ rf
    PR✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    CO✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    PC✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    RF✝ : α → Nat.Partrec.Code → σ → σ := fun a cf hf => rf a { fst := cf, snd :=  …
    F : α → Nat.Partrec.Code → σ := fun a c => Nat.Partrec.Code.recOn c (z a) (s a …
    G₁ : Prod (Prod α (List σ)) (Prod Nat Nat) → Option σ := fun p => (p.1.2.get?  …
    this✝ : Primrec G₁
    G : α → List σ → Option σ := fun a IH => Nat.casesOn IH.length (Option.some (z …
    this : Primrec₂ G
    a : α
    n : Nat
    m : Nat := n.div2.div2
    hm : LT.lt m (HAdd.hAdd n 4)
    m1 : LT.lt (Nat.unpair m).1 (HAdd.hAdd n 4)
    m2 : LT.lt (Nat.unpair m).2 (HAdd.hAdd n 4)
    ⊢ Eq (cond n.bodd (cond n.div2.bodd (rf a { fst := Denumerable.ofNat Nat.Partr …
  -/
  simp [ofNatCode]
  /-
    case succ.succ.succ.succ
    α : Type u_1
    σ : Type u_2
    inst✝¹ : Primcodable α
    inst✝ : Primcodable σ
    c : α → Nat.Partrec.Code
    hc : Primrec c
    z : α → σ
    hz : Primrec z
    s : α → σ
    hs : Primrec s
    l : α → σ
    hl : Primrec l
    r : α → σ
    hr : Primrec r
    pr : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpr : Primrec₂ pr
    co : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hco : Primrec₂ co
    pc : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpc : Primrec₂ pc
    rf : α → Prod Nat.Partrec.Code σ → σ
    hrf : Primrec₂ rf
    PR✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    CO✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    PC✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    RF✝ : α → Nat.Partrec.Code → σ → σ := fun a cf hf => rf a { fst := cf, snd :=  …
    F : α → Nat.Partrec.Code → σ := fun a c => Nat.Partrec.Code.recOn c (z a) (s a …
    G₁ : Prod (Prod α (List σ)) (Prod Nat Nat) → Option σ := fun p => (p.1.2.get?  …
    this✝ : Primrec G₁
    G : α → List σ → Option σ := fun a IH => Nat.casesOn IH.length (Option.some (z …
    this : Primrec₂ G
    a : α
    n : Nat
    m : Nat := n.div2.div2
    hm : LT.lt m (HAdd.hAdd n 4)
    m1 : LT.lt (Nat.unpair m).1 (HAdd.hAdd n 4)
    m2 : LT.lt (Nat.unpair m).2 (HAdd.hAdd n 4)
    ⊢ Eq (cond n.bodd (cond n.div2.bodd (rf a { fst := Denumerable.ofNat Nat.Partr …
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
  cases n.bodd <;> cases n.div2.bodd <;> rfl
                                         /-
                                           🎉 no goals
                                         -/


/-- Recursion on `Nat.Partrec.Code` is primitive recursive. -/
theorem rec_prim {α σ} [Primcodable α] [Primcodable σ] {c : α → Code} (hc : Primrec c) {z : α → σ}
    (hz : Primrec z) {s : α → σ} (hs : Primrec s) {l : α → σ} (hl : Primrec l) {r : α → σ}
    (hr : Primrec r) {pr : α → Code → Code → σ → σ → σ}
    (hpr : Primrec fun a : α × Code × Code × σ × σ => pr a.1 a.2.1 a.2.2.1 a.2.2.2.1 a.2.2.2.2)
    {co : α → Code → Code → σ → σ → σ}
    (hco : Primrec fun a : α × Code × Code × σ × σ => co a.1 a.2.1 a.2.2.1 a.2.2.2.1 a.2.2.2.2)
    {pc : α → Code → Code → σ → σ → σ}
    (hpc : Primrec fun a : α × Code × Code × σ × σ => pc a.1 a.2.1 a.2.2.1 a.2.2.2.1 a.2.2.2.2)
    {rf : α → Code → σ → σ} (hrf : Primrec fun a : α × Code × σ => rf a.1 a.2.1 a.2.2) :
    let F (a : α) (c : Code) : σ :=
      Nat.Partrec.Code.recOn c (z a) (s a) (l a) (r a) (pr a) (co a) (pc a) (rf a)
    Primrec fun a => F a (c a) :=
  rec_prim' hc hz hs hl hr
    (pr := fun a b => pr a b.1 b.2.1 b.2.2.1 b.2.2.2) (.mk hpr)
    (co := fun a b => co a b.1 b.2.1 b.2.2.1 b.2.2.2) (.mk hco)
    (pc := fun a b => pc a b.1 b.2.1 b.2.2.1 b.2.2.2) (.mk hpc)
    (rf := fun a b => rf a b.1 b.2) (.mk hrf)


/-- Recursion on `Nat.Partrec.Code` is computable. -/
theorem rec_computable {α σ} [Primcodable α] [Primcodable σ] {c : α → Code} (hc : Computable c)
    {z : α → σ} (hz : Computable z) {s : α → σ} (hs : Computable s) {l : α → σ} (hl : Computable l)
    {r : α → σ} (hr : Computable r) {pr : α → Code × Code × σ × σ → σ} (hpr : Computable₂ pr)
    {co : α → Code × Code × σ × σ → σ} (hco : Computable₂ co) {pc : α → Code × Code × σ × σ → σ}
    (hpc : Computable₂ pc) {rf : α → Code × σ → σ} (hrf : Computable₂ rf) :
    let PR (a) cf cg hf hg := pr a (cf, cg, hf, hg)
    let CO (a) cf cg hf hg := co a (cf, cg, hf, hg)
    let PC (a) cf cg hf hg := pc a (cf, cg, hf, hg)
    let RF (a) cf hf := rf a (cf, hf)
    let F (a : α) (c : Code) : σ :=
      Nat.Partrec.Code.recOn c (z a) (s a) (l a) (r a) (PR a) (CO a) (PC a) (RF a)
    Computable fun a => F a (c a) := by
  -- TODO(Mario): less copy-paste from previous proof
  /-
    α : Type u_1
    σ : Type u_2
    inst✝¹ : Primcodable α
    inst✝ : Primcodable σ
    c : α → Nat.Partrec.Code
    hc : Computable c
    z : α → σ
    hz : Computable z
    s : α → σ
    hs : Computable s
    l : α → σ
    hl : Computable l
    r : α → σ
    hr : Computable r
    pr : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpr : Computable₂ pr
    co : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hco : Computable₂ co
    pc : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpc : Computable₂ pc
    rf : α → Prod Nat.Partrec.Code σ → σ
    hrf : Computable₂ rf
    ⊢ let PR := fun a cf cg hf hg => pr a { fst := cf, snd := { fst := cg, snd :=  …
      let CO := fun a cf cg hf hg => co a { fst := cf, snd := { fst := cg, snd :=  …
      let PC := fun a cf cg hf hg => pc a { fst := cf, snd := { fst := cg, snd :=  …
      let RF := fun a cf hf => rf a { fst := cf, snd := hf };
      let F := fun a c => Nat.Partrec.Code.recOn c (z a) (s a) (l a) (r a) (PR a)  …
      Computable fun a => F a (c a)
  -/
  intros _ _ _ _ F
  let G₁ : (α × List σ) × ℕ × ℕ → Option σ := fun p =>
    letI a := p.1.1; letI IH := p.1.2; letI n := p.2.1; letI m := p.2.2
    (IH.get? m).bind fun s =>
    (IH.get? m.unpair.1).bind fun s₁ =>
    (IH.get? m.unpair.2).map fun s₂ =>
    cond n.bodd
      (cond n.div2.bodd (rf a (ofNat Code m, s))
        (pc a (ofNat Code m.unpair.1, ofNat Code m.unpair.2, s₁, s₂)))
      (cond n.div2.bodd (co a (ofNat Code m.unpair.1, ofNat Code m.unpair.2, s₁, s₂))
        (pr a (ofNat Code m.unpair.1, ofNat Code m.unpair.2, s₁, s₂)))
  have : Computable G₁ := by
    refine option_bind (list_get?.comp (snd.comp fst) (snd.comp snd)) <| .mk ?_
    refine option_bind ((list_get?.comp (snd.comp fst)
      (fst.comp <| Computable.unpair.comp (snd.comp snd))).comp fst) <| .mk ?_
    refine option_map ((list_get?.comp (snd.comp fst)
      (snd.comp <| Computable.unpair.comp (snd.comp snd))).comp <| fst.comp fst) <| .mk ?_
    exact
      have a := fst.comp (fst.comp <| fst.comp <| fst.comp fst)
      have n := fst.comp (snd.comp <| fst.comp <| fst.comp fst)
      have m := snd.comp (snd.comp <| fst.comp <| fst.comp fst)
      have m₁ := fst.comp (Computable.unpair.comp m)
      have m₂ := snd.comp (Computable.unpair.comp m)
      have s := snd.comp (fst.comp fst)
      have s₁ := snd.comp fst
      have s₂ := snd
      (nat_bodd.comp n).cond
        ((nat_bodd.comp <| nat_div2.comp n).cond
          (hrf.comp a (((Computable.ofNat Code).comp m).pair s))
          (hpc.comp a (((Computable.ofNat Code).comp m₁).pair <|
            ((Computable.ofNat Code).comp m₂).pair <| s₁.pair s₂)))
        (Computable.cond (nat_bodd.comp <| nat_div2.comp n)
          (hco.comp a (((Computable.ofNat Code).comp m₁).pair <|
            ((Computable.ofNat Code).comp m₂).pair <| s₁.pair s₂))
          (hpr.comp a (((Computable.ofNat Code).comp m₁).pair <|
            ((Computable.ofNat Code).comp m₂).pair <| s₁.pair s₂)))
  let G : α → List σ → Option σ := fun a IH =>
    IH.length.casesOn (some (z a)) fun n =>
    n.casesOn (some (s a)) fun n =>
    n.casesOn (some (l a)) fun n =>
    n.casesOn (some (r a)) fun n =>
    G₁ ((a, IH), n, n.div2.div2)
  have : Computable₂ G := .mk <|
    nat_casesOn (list_length.comp snd) (option_some_iff.2 (hz.comp fst)) <| .mk <|
    nat_casesOn snd (option_some_iff.2 (hs.comp (fst.comp fst))) <| .mk <|
    nat_casesOn snd (option_some_iff.2 (hl.comp (fst.comp <| fst.comp fst))) <| .mk <|
    nat_casesOn snd (option_some_iff.2 (hr.comp (fst.comp <| fst.comp <| fst.comp fst))) <| .mk <|
    this.comp <|
      ((fst.pair snd).comp <| fst.comp <| fst.comp <| fst.comp <| fst).pair <|
      snd.pair <| nat_div2.comp <| nat_div2.comp snd
  refine (nat_strong_rec (fun a n => F a (ofNat Code n)) this.to₂ fun a n => ?_)
    |>.comp .id (encode_iff.2 hc) |>.of_eq fun a => by simp
  /-
    α : Type u_1
    σ : Type u_2
    inst✝¹ : Primcodable α
    inst✝ : Primcodable σ
    c : α → Nat.Partrec.Code
    hc : Computable c
    z : α → σ
    hz : Computable z
    s : α → σ
    hs : Computable s
    l : α → σ
    hl : Computable l
    r : α → σ
    hr : Computable r
    pr : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpr : Computable₂ pr
    co : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hco : Computable₂ co
    pc : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpc : Computable₂ pc
    rf : α → Prod Nat.Partrec.Code σ → σ
    hrf : Computable₂ rf
    PR✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    CO✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    PC✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    RF✝ : α → Nat.Partrec.Code → σ → σ := fun a cf hf => rf a { fst := cf, snd :=  …
    F : α → Nat.Partrec.Code → σ := fun a c => Nat.Partrec.Code.recOn c (z a) (s a …
    G₁ : Prod (Prod α (List σ)) (Prod Nat Nat) → Option σ := fun p => (p.1.2.get?  …
    this✝ : Computable G₁
    G : α → List σ → Option σ := fun a IH => Nat.casesOn IH.length (Option.some (z …
    this : Computable₂ G
    a : α
    n : Nat
    ⊢ Eq (G { fst := a, snd := List.map ((fun a n => F a (Denumerable.ofNat Nat.Pa …
  -/
  iterate 4 cases' n with n; · simp [ofNatCode_eq, ofNatCode]; rfl
  /-
    case succ.succ.succ.succ
    α : Type u_1
    σ : Type u_2
    inst✝¹ : Primcodable α
    inst✝ : Primcodable σ
    c : α → Nat.Partrec.Code
    hc : Computable c
    z : α → σ
    hz : Computable z
    s : α → σ
    hs : Computable s
    l : α → σ
    hl : Computable l
    r : α → σ
    hr : Computable r
    pr : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpr : Computable₂ pr
    co : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hco : Computable₂ co
    pc : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpc : Computable₂ pc
    rf : α → Prod Nat.Partrec.Code σ → σ
    hrf : Computable₂ rf
    PR✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    CO✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    PC✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    RF✝ : α → Nat.Partrec.Code → σ → σ := fun a cf hf => rf a { fst := cf, snd :=  …
    F : α → Nat.Partrec.Code → σ := fun a c => Nat.Partrec.Code.recOn c (z a) (s a …
    G₁ : Prod (Prod α (List σ)) (Prod Nat Nat) → Option σ := fun p => (p.1.2.get?  …
    this✝ : Computable G₁
    G : α → List σ → Option σ := fun a IH => Nat.casesOn IH.length (Option.some (z …
    this : Computable₂ G
    a : α
    n : Nat
    ⊢ Eq (G { fst := a, snd := List.map ((fun a n => F a (Denumerable.ofNat Nat.Pa …
  -/
  simp only [G]; rw [List.length_map, List.length_range]
  /-
    case succ.succ.succ.succ
    α : Type u_1
    σ : Type u_2
    inst✝¹ : Primcodable α
    inst✝ : Primcodable σ
    c : α → Nat.Partrec.Code
    hc : Computable c
    z : α → σ
    hz : Computable z
    s : α → σ
    hs : Computable s
    l : α → σ
    hl : Computable l
    r : α → σ
    hr : Computable r
    pr : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpr : Computable₂ pr
    co : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hco : Computable₂ co
    pc : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpc : Computable₂ pc
    rf : α → Prod Nat.Partrec.Code σ → σ
    hrf : Computable₂ rf
    PR✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    CO✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    PC✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    RF✝ : α → Nat.Partrec.Code → σ → σ := fun a cf hf => rf a { fst := cf, snd :=  …
    F : α → Nat.Partrec.Code → σ := fun a c => Nat.Partrec.Code.recOn c (z a) (s a …
    G₁ : Prod (Prod α (List σ)) (Prod Nat Nat) → Option σ := fun p => (p.1.2.get?  …
    this✝ : Computable G₁
    G : α → List σ → Option σ := fun a IH => Nat.casesOn IH.length (Option.some (z …
    this : Computable₂ G
    a : α
    n : Nat
    ⊢ Eq (Nat.rec (Option.some (z a)) (fun n_1 n_ih => Nat.rec (Option.some (s a)) …
  -/
  let m := n.div2.div2
  show G₁ ((a, (List.range (n + 4)).map fun n => F a (ofNat Code n)), n, m)
    = some (F a (ofNat Code (n + 4)))
  have hm : m < n + 4 := by
    simp only [m, div2_val]
    exact lt_of_le_of_lt
      (le_trans (Nat.div_le_self ..) (Nat.div_le_self ..))
      (Nat.succ_le_succ (Nat.le_add_right ..))
  /-
    case succ.succ.succ.succ
    α : Type u_1
    σ : Type u_2
    inst✝¹ : Primcodable α
    inst✝ : Primcodable σ
    c : α → Nat.Partrec.Code
    hc : Computable c
    z : α → σ
    hz : Computable z
    s : α → σ
    hs : Computable s
    l : α → σ
    hl : Computable l
    r : α → σ
    hr : Computable r
    pr : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpr : Computable₂ pr
    co : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hco : Computable₂ co
    pc : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpc : Computable₂ pc
    rf : α → Prod Nat.Partrec.Code σ → σ
    hrf : Computable₂ rf
    PR✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    CO✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    PC✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    RF✝ : α → Nat.Partrec.Code → σ → σ := fun a cf hf => rf a { fst := cf, snd :=  …
    F : α → Nat.Partrec.Code → σ := fun a c => Nat.Partrec.Code.recOn c (z a) (s a …
    G₁ : Prod (Prod α (List σ)) (Prod Nat Nat) → Option σ := fun p => (p.1.2.get?  …
    this✝ : Computable G₁
    G : α → List σ → Option σ := fun a IH => Nat.casesOn IH.length (Option.some (z …
    this : Computable₂ G
    a : α
    n : Nat
    m : Nat := n.div2.div2
    hm : LT.lt m (HAdd.hAdd n 4)
    ⊢ Eq (G₁ { fst := { fst := a, snd := List.map (fun n => F a (Denumerable.ofNat …
  -/
  have m1 : m.unpair.1 < n + 4 := lt_of_le_of_lt m.unpair_left_le hm
  /-
    case succ.succ.succ.succ
    α : Type u_1
    σ : Type u_2
    inst✝¹ : Primcodable α
    inst✝ : Primcodable σ
    c : α → Nat.Partrec.Code
    hc : Computable c
    z : α → σ
    hz : Computable z
    s : α → σ
    hs : Computable s
    l : α → σ
    hl : Computable l
    r : α → σ
    hr : Computable r
    pr : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpr : Computable₂ pr
    co : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hco : Computable₂ co
    pc : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpc : Computable₂ pc
    rf : α → Prod Nat.Partrec.Code σ → σ
    hrf : Computable₂ rf
    PR✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    CO✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    PC✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    RF✝ : α → Nat.Partrec.Code → σ → σ := fun a cf hf => rf a { fst := cf, snd :=  …
    F : α → Nat.Partrec.Code → σ := fun a c => Nat.Partrec.Code.recOn c (z a) (s a …
    G₁ : Prod (Prod α (List σ)) (Prod Nat Nat) → Option σ := fun p => (p.1.2.get?  …
    this✝ : Computable G₁
    G : α → List σ → Option σ := fun a IH => Nat.casesOn IH.length (Option.some (z …
    this : Computable₂ G
    a : α
    n : Nat
    m : Nat := n.div2.div2
    hm : LT.lt m (HAdd.hAdd n 4)
    m1 : LT.lt (Nat.unpair m).1 (HAdd.hAdd n 4)
    ⊢ Eq (G₁ { fst := { fst := a, snd := List.map (fun n => F a (Denumerable.ofNat …
  -/
  have m2 : m.unpair.2 < n + 4 := lt_of_le_of_lt m.unpair_right_le hm
  /-
    case succ.succ.succ.succ
    α : Type u_1
    σ : Type u_2
    inst✝¹ : Primcodable α
    inst✝ : Primcodable σ
    c : α → Nat.Partrec.Code
    hc : Computable c
    z : α → σ
    hz : Computable z
    s : α → σ
    hs : Computable s
    l : α → σ
    hl : Computable l
    r : α → σ
    hr : Computable r
    pr : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpr : Computable₂ pr
    co : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hco : Computable₂ co
    pc : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpc : Computable₂ pc
    rf : α → Prod Nat.Partrec.Code σ → σ
    hrf : Computable₂ rf
    PR✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    CO✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    PC✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    RF✝ : α → Nat.Partrec.Code → σ → σ := fun a cf hf => rf a { fst := cf, snd :=  …
    F : α → Nat.Partrec.Code → σ := fun a c => Nat.Partrec.Code.recOn c (z a) (s a …
    G₁ : Prod (Prod α (List σ)) (Prod Nat Nat) → Option σ := fun p => (p.1.2.get?  …
    this✝ : Computable G₁
    G : α → List σ → Option σ := fun a IH => Nat.casesOn IH.length (Option.some (z …
    this : Computable₂ G
    a : α
    n : Nat
    m : Nat := n.div2.div2
    hm : LT.lt m (HAdd.hAdd n 4)
    m1 : LT.lt (Nat.unpair m).1 (HAdd.hAdd n 4)
    m2 : LT.lt (Nat.unpair m).2 (HAdd.hAdd n 4)
    ⊢ Eq (G₁ { fst := { fst := a, snd := List.map (fun n => F a (Denumerable.ofNat …
  -/
  simp [G₁, m, List.getElem?_map, List.getElem?_range, hm, m1, m2]
  /-
    case succ.succ.succ.succ
    α : Type u_1
    σ : Type u_2
    inst✝¹ : Primcodable α
    inst✝ : Primcodable σ
    c : α → Nat.Partrec.Code
    hc : Computable c
    z : α → σ
    hz : Computable z
    s : α → σ
    hs : Computable s
    l : α → σ
    hl : Computable l
    r : α → σ
    hr : Computable r
    pr : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpr : Computable₂ pr
    co : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hco : Computable₂ co
    pc : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpc : Computable₂ pc
    rf : α → Prod Nat.Partrec.Code σ → σ
    hrf : Computable₂ rf
    PR✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    CO✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    PC✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    RF✝ : α → Nat.Partrec.Code → σ → σ := fun a cf hf => rf a { fst := cf, snd :=  …
    F : α → Nat.Partrec.Code → σ := fun a c => Nat.Partrec.Code.recOn c (z a) (s a …
    G₁ : Prod (Prod α (List σ)) (Prod Nat Nat) → Option σ := fun p => (p.1.2.get?  …
    this✝ : Computable G₁
    G : α → List σ → Option σ := fun a IH => Nat.casesOn IH.length (Option.some (z …
    this : Computable₂ G
    a : α
    n : Nat
    m : Nat := n.div2.div2
    hm : LT.lt m (HAdd.hAdd n 4)
    m1 : LT.lt (Nat.unpair m).1 (HAdd.hAdd n 4)
    m2 : LT.lt (Nat.unpair m).2 (HAdd.hAdd n 4)
    ⊢ Eq (cond n.bodd (cond n.div2.bodd (rf a { fst := Denumerable.ofNat Nat.Partr …
  -/
  rw [show ofNat Code (n + 4) = ofNatCode (n + 4) from rfl]
  /-
    case succ.succ.succ.succ
    α : Type u_1
    σ : Type u_2
    inst✝¹ : Primcodable α
    inst✝ : Primcodable σ
    c : α → Nat.Partrec.Code
    hc : Computable c
    z : α → σ
    hz : Computable z
    s : α → σ
    hs : Computable s
    l : α → σ
    hl : Computable l
    r : α → σ
    hr : Computable r
    pr : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpr : Computable₂ pr
    co : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hco : Computable₂ co
    pc : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpc : Computable₂ pc
    rf : α → Prod Nat.Partrec.Code σ → σ
    hrf : Computable₂ rf
    PR✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    CO✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    PC✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    RF✝ : α → Nat.Partrec.Code → σ → σ := fun a cf hf => rf a { fst := cf, snd :=  …
    F : α → Nat.Partrec.Code → σ := fun a c => Nat.Partrec.Code.recOn c (z a) (s a …
    G₁ : Prod (Prod α (List σ)) (Prod Nat Nat) → Option σ := fun p => (p.1.2.get?  …
    this✝ : Computable G₁
    G : α → List σ → Option σ := fun a IH => Nat.casesOn IH.length (Option.some (z …
    this : Computable₂ G
    a : α
    n : Nat
    m : Nat := n.div2.div2
    hm : LT.lt m (HAdd.hAdd n 4)
    m1 : LT.lt (Nat.unpair m).1 (HAdd.hAdd n 4)
    m2 : LT.lt (Nat.unpair m).2 (HAdd.hAdd n 4)
    ⊢ Eq (cond n.bodd (cond n.div2.bodd (rf a { fst := Denumerable.ofNat Nat.Partr …
  -/
  simp [ofNatCode]
  /-
    case succ.succ.succ.succ
    α : Type u_1
    σ : Type u_2
    inst✝¹ : Primcodable α
    inst✝ : Primcodable σ
    c : α → Nat.Partrec.Code
    hc : Computable c
    z : α → σ
    hz : Computable z
    s : α → σ
    hs : Computable s
    l : α → σ
    hl : Computable l
    r : α → σ
    hr : Computable r
    pr : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpr : Computable₂ pr
    co : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hco : Computable₂ co
    pc : α → Prod Nat.Partrec.Code (Prod Nat.Partrec.Code (Prod σ σ)) → σ
    hpc : Computable₂ pc
    rf : α → Prod Nat.Partrec.Code σ → σ
    hrf : Computable₂ rf
    PR✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    CO✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    PC✝ : α → Nat.Partrec.Code → Nat.Partrec.Code → σ → σ → σ := fun a cf cg hf hg …
    RF✝ : α → Nat.Partrec.Code → σ → σ := fun a cf hf => rf a { fst := cf, snd :=  …
    F : α → Nat.Partrec.Code → σ := fun a c => Nat.Partrec.Code.recOn c (z a) (s a …
    G₁ : Prod (Prod α (List σ)) (Prod Nat Nat) → Option σ := fun p => (p.1.2.get?  …
    this✝ : Computable G₁
    G : α → List σ → Option σ := fun a IH => Nat.casesOn IH.length (Option.some (z …
    this : Computable₂ G
    a : α
    n : Nat
    m : Nat := n.div2.div2
    hm : LT.lt m (HAdd.hAdd n 4)
    m1 : LT.lt (Nat.unpair m).1 (HAdd.hAdd n 4)
    m2 : LT.lt (Nat.unpair m).2 (HAdd.hAdd n 4)
    ⊢ Eq (cond n.bodd (cond n.div2.bodd (rf a { fst := Denumerable.ofNat Nat.Partr …
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
  cases n.bodd <;> cases n.div2.bodd <;> rfl
                                         /-
                                           🎉 no goals
                                         -/


/-- The interpretation of a `Nat.Partrec.Code` as a partial function.
* `Nat.Partrec.Code.zero`: The constant zero function.
* `Nat.Partrec.Code.succ`: The successor function.
* `Nat.Partrec.Code.left`: Left unpairing of a pair of ℕ (encoded by `Nat.pair`)
* `Nat.Partrec.Code.right`: Right unpairing of a pair of ℕ (encoded by `Nat.pair`)
* `Nat.Partrec.Code.pair`: Pairs the outputs of argument codes using `Nat.pair`.
* `Nat.Partrec.Code.comp`: Composition of two argument codes.
* `Nat.Partrec.Code.prec`: Primitive recursion. Given an argument of the form `Nat.pair a n`:
  * If `n = 0`, returns `eval cf a`.
  * If `n = succ k`, returns `eval cg (pair a (pair k (eval (prec cf cg) (pair a k))))`
* `Nat.Partrec.Code.rfind'`: Minimization. For `f` an argument of the form `Nat.pair a m`,
  `rfind' f m` returns the least `a` such that `f a m = 0`, if one exists and `f b m` terminates
  for `b < a`
-/
def eval : Code → ℕ →. ℕ
  | zero => pure 0
  | succ => Nat.succ
  | left => ↑fun n : ℕ => n.unpair.1
  | right => ↑fun n : ℕ => n.unpair.2
  | pair cf cg => fun n => Nat.pair <$> eval cf n <*> eval cg n
  | comp cf cg => fun n => eval cg n >>= eval cf
  | prec cf cg =>
    Nat.unpaired fun a n =>
      n.rec (eval cf a) fun y IH => do
        let i ← IH
        eval cg (Nat.pair a (Nat.pair y i))
  | rfind' cf =>
    Nat.unpaired fun a m =>
      (Nat.rfind fun n => (fun m => m = 0) <$> eval cf (Nat.pair a (n + m))).map (· + m)


/-- Helper lemma for the evaluation of `prec` in the base case. -/
@[simp]
theorem eval_prec_zero (cf cg : Code) (a : ℕ) : eval (prec cf cg) (Nat.pair a 0) = eval cf a := by
  /-
    cf cg : Nat.Partrec.Code
    a : Nat
    ⊢ Eq ((cf.prec cg).eval (Nat.pair a 0)) (cf.eval a)
  -/
  rw [eval, Nat.unpaired, Nat.unpair_pair]
  /-
    cf cg : Nat.Partrec.Code
    a : Nat
    ⊢ Eq (Nat.rec (cf.eval { fst := a, snd := 0 }.1) (fun y IH => Bind.bind IH fun …
  -/
  simp (config := { Lean.Meta.Simp.neutralConfig with proj := true }) only []
  /-
    cf cg : Nat.Partrec.Code
    a : Nat
    ⊢ Eq (Nat.rec (cf.eval a) (fun y IH => Bind.bind IH fun i => cg.eval (Nat.pair …
  -/
  rw [Nat.rec_zero]
  /-
    🎉 no goals
  -/


/-- Helper lemma for the evaluation of `prec` in the recursive case. -/
theorem eval_prec_succ (cf cg : Code) (a k : ℕ) :
    eval (prec cf cg) (Nat.pair a (Nat.succ k)) =
      do {let ih ← eval (prec cf cg) (Nat.pair a k); eval cg (Nat.pair a (Nat.pair k ih))} := by
  /-
    cf cg : Nat.Partrec.Code
    a k : Nat
    ⊢ Eq ((cf.prec cg).eval (Nat.pair a k.succ)) (Bind.bind ((cf.prec cg).eval (Na …
  -/
  rw [eval, Nat.unpaired, Part.bind_eq_bind, Nat.unpair_pair]
  /-
    cf cg : Nat.Partrec.Code
    a k : Nat
    ⊢ Eq (Nat.rec (cf.eval { fst := a, snd := k.succ }.1) (fun y IH => Bind.bind I …
  -/
  simp
  /-
    🎉 no goals
  -/


instance : Membership (ℕ →. ℕ) Code :=
  ⟨fun c f => eval c = f⟩


@[simp]
theorem eval_const : ∀ n m, eval (Code.const n) m = Part.some n
  | 0, _ => rfl
                   /-
                     n m : Nat
                     ⊢ Eq ((Nat.Partrec.Code.const (HAdd.hAdd n 1)).eval m) (Part.some (HAdd.hAdd n …
                   -/
  | n + 1, m => by simp! [eval_const n m]
                   /-
                     🎉 no goals
                   -/


@[simp]
                                                         /-
                                                           n : Nat
                                                           ⊢ Eq (Nat.Partrec.Code.id.eval n) (Part.some n)
                                                         -/
theorem eval_id (n) : eval Code.id n = Part.some n := by simp! [Seq.seq, Code.id]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp]
                                                                              /-
                                                                                c : Nat.Partrec.Code
                                                                                n x : Nat
                                                                                ⊢ Eq ((c.curry n).eval x) (c.eval (Nat.pair n x))
                                                                              -/
theorem eval_curry (c n x) : eval (curry c n) x = eval c (Nat.pair n x) := by simp! [Seq.seq, curry]
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


theorem const_prim : Primrec Code.const :=
  (_root_.Primrec.id.nat_iterate (_root_.Primrec.const zero)
    (comp_prim.comp (_root_.Primrec.const succ) Primrec.snd).to₂).of_eq
                /-
                  n : Nat
                  ⊢ Eq (Nat.iterate (fun b => Nat.Partrec.Code.succ.comp { fst := n, snd := b }. …
                -/
    fun n => by simp; induction n <;>
      /-
        case zero
        ⊢ Eq (Nat.iterate (fun b => Nat.Partrec.Code.succ.comp b) 0 Nat.Partrec.Code.z …
      -/
      /-
        🎉 no goals
      -/
      simp [*, Code.const, Function.iterate_succ', -Function.iterate_succ]
      /-
        🎉 no goals
      -/


theorem curry_prim : Primrec₂ curry :=
  comp_prim.comp Primrec.fst <| pair_prim.comp (const_prim.comp Primrec.snd)
    (_root_.Primrec.const Code.id)


theorem curry_inj {c₁ c₂ n₁ n₂} (h : curry c₁ n₁ = curry c₂ n₂) : c₁ = c₂ ∧ n₁ = n₂ :=
      /-
        c₁ c₂ : Nat.Partrec.Code
        n₁ n₂ : Nat
        h : Eq (c₁.curry n₁) (c₂.curry n₂)
        ⊢ Eq c₁ c₂
      -/
  ⟨by injection h, by
      /-
        🎉 no goals
      -/
    /-
      c₁ c₂ : Nat.Partrec.Code
      n₁ n₂ : Nat
      h : Eq (c₁.curry n₁) (c₂.curry n₂)
      ⊢ Eq n₁ n₂
    -/
    injection h with h₁ h₂
    /-
      c₁ c₂ : Nat.Partrec.Code
      n₁ n₂ : Nat
      h₁ : Eq c₁ c₂
      h₂ : Eq ((Nat.Partrec.Code.const n₁).pair Nat.Partrec.Code.id) ((Nat.Partrec.C …
      ⊢ Eq n₁ n₂
    -/
    injection h₂ with h₃ h₄
    /-
      c₁ c₂ : Nat.Partrec.Code
      n₁ n₂ : Nat
      h₁ : Eq c₁ c₂
      h₃ : Eq (Nat.Partrec.Code.const n₁) (Nat.Partrec.Code.const n₂)
      h₄ : Eq Nat.Partrec.Code.id Nat.Partrec.Code.id
      ⊢ Eq n₁ n₂
    -/
    exact const_inj h₃⟩
    /-
      🎉 no goals
    -/


/--
The $S_n^m$ theorem: There is a computable function, namely `Nat.Partrec.Code.curry`, that takes a
program and a ℕ `n`, and returns a new program using `n` as the first argument.
-/
theorem smn :
    ∃ f : Code → ℕ → Code, Computable₂ f ∧ ∀ c n x, eval (f c n) x = eval c (Nat.pair n x) :=
  ⟨curry, Primrec₂.to_comp curry_prim, eval_curry⟩


/-- A function is partial recursive if and only if there is a code implementing it. Therefore,
`eval` is a **universal partial recursive function**. -/
theorem exists_code {f : ℕ →. ℕ} : Nat.Partrec f ↔ ∃ c : Code, eval c = f := by
  /-
    f : PFun Nat Nat
    ⊢ Iff (Nat.Partrec f) (Exists fun c => Eq c.eval f)
  -/
  refine ⟨fun h => ?_, ?_⟩
  · induction h with
    | zero => exact ⟨zero, rfl⟩
    | succ => exact ⟨succ, rfl⟩
    | left => exact ⟨left, rfl⟩
    | right => exact ⟨right, rfl⟩
    | pair pf pg hf hg =>
      rcases hf with ⟨cf, rfl⟩; rcases hg with ⟨cg, rfl⟩
      exact ⟨pair cf cg, rfl⟩
    | comp pf pg hf hg =>
      rcases hf with ⟨cf, rfl⟩; rcases hg with ⟨cg, rfl⟩
      exact ⟨comp cf cg, rfl⟩
    | prec pf pg hf hg =>
      rcases hf with ⟨cf, rfl⟩; rcases hg with ⟨cg, rfl⟩
      exact ⟨prec cf cg, rfl⟩
    | rfind pf hf =>
      rcases hf with ⟨cf, rfl⟩
      refine ⟨comp (rfind' cf) (pair Code.id zero), ?_⟩
      simp [eval, Seq.seq, pure, PFun.pure, Part.map_id']
    /-
      case refine_2
      f : PFun Nat Nat
      ⊢ (Exists fun c => Eq c.eval f) → Nat.Partrec f
    -/
  · rintro ⟨c, rfl⟩
    induction c with
    | zero => exact Nat.Partrec.zero
    | succ => exact Nat.Partrec.succ
    | left => exact Nat.Partrec.left
    | right => exact Nat.Partrec.right
    | pair cf cg pf pg => exact pf.pair pg
    | comp cf cg pf pg => exact pf.comp pg
    | prec cf cg pf pg => exact pf.prec pg
    | rfind' cf pf => exact pf.rfind'

-- Porting note: `>>`s in `evaln` are now `>>=` because `>>`s are not elaborated well in Lean4.

/-- A modified evaluation for the code which returns an `Option ℕ` instead of a `Part ℕ`. To avoid
undecidability, `evaln` takes a parameter `k` and fails if it encounters a number ≥ k in the course
of its execution. Other than this, the semantics are the same as in `Nat.Partrec.Code.eval`.
-/
def evaln : ℕ → Code → ℕ → Option ℕ
  | 0, _ => fun _ => Option.none
  | k + 1, zero => fun n => do
    guard (n ≤ k)
    return 0
  | k + 1, succ => fun n => do
    guard (n ≤ k)
    return (Nat.succ n)
  | k + 1, left => fun n => do
    guard (n ≤ k)
    return n.unpair.1
  | k + 1, right => fun n => do
    guard (n ≤ k)
    pure n.unpair.2
  | k + 1, pair cf cg => fun n => do
    guard (n ≤ k)
    Nat.pair <$> evaln (k + 1) cf n <*> evaln (k + 1) cg n
  | k + 1, comp cf cg => fun n => do
    guard (n ≤ k)
    let x ← evaln (k + 1) cg n
    evaln (k + 1) cf x
  | k + 1, prec cf cg => fun n => do
    guard (n ≤ k)
    n.unpaired fun a n =>
      n.casesOn (evaln (k + 1) cf a) fun y => do
        let i ← evaln k (prec cf cg) (Nat.pair a y)
        evaln (k + 1) cg (Nat.pair a (Nat.pair y i))
  | k + 1, rfind' cf => fun n => do
    guard (n ≤ k)
    n.unpaired fun a m => do
      let x ← evaln (k + 1) cf (Nat.pair a m)
      if x = 0 then
        pure m
      else
        evaln k (rfind' cf) (Nat.pair a (m + 1))


theorem evaln_bound : ∀ {k c n x}, x ∈ evaln k c n → n < k
                        /-
                          c : Nat.Partrec.Code
                          n x : Nat
                          h : Membership.mem (Nat.Partrec.Code.evaln 0 c n) x
                          ⊢ LT.lt n 0
                        -/
  | 0, c, n, x, h => by simp [evaln] at h
                        /-
                          🎉 no goals
                        -/
  | k + 1, c, n, x, h => by
    suffices ∀ {o : Option ℕ}, x ∈ do { guard (n ≤ k); o } → n < k + 1 by
      cases c <;> rw [evaln] at h <;> exact this h
    /-
      k : Nat
      c : Nat.Partrec.Code
      n x : Nat
      h : Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) c n) x
      ⊢ ∀ {o : Option Nat}, Membership.mem (Bind.bind (guard (LE.le n k)) fun x => o …
    -/
    simpa [Option.bind_eq_some] using Nat.lt_succ_of_le
    /-
      🎉 no goals
    -/


theorem evaln_mono : ∀ {k₁ k₂ c n x}, k₁ ≤ k₂ → x ∈ evaln k₁ c n → x ∈ evaln k₂ c n
                               /-
                                 k₂ : Nat
                                 c : Nat.Partrec.Code
                                 n x : Nat
                                 x✝ : LE.le 0 k₂
                                 h : Membership.mem (Nat.Partrec.Code.evaln 0 c n) x
                                 ⊢ Membership.mem (Nat.Partrec.Code.evaln k₂ c n) x
                               -/
  | 0, k₂, c, n, x, _, h => by simp [evaln] at h
                               /-
                                 🎉 no goals
                               -/
  | k + 1, k₂ + 1, c, n, x, hl, h => by
    /-
      k k₂ : Nat
      c : Nat.Partrec.Code
      n x : Nat
      hl : LE.le (HAdd.hAdd k 1) (HAdd.hAdd k₂ 1)
      h : Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) c n) x
      ⊢ Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k₂ 1) c n) x
    -/
    have hl' := Nat.le_of_succ_le_succ hl
    have :
      ∀ {k k₂ n x : ℕ} {o₁ o₂ : Option ℕ},
        k ≤ k₂ → (x ∈ o₁ → x ∈ o₂) →
          x ∈ do { guard (n ≤ k); o₁ } → x ∈ do { guard (n ≤ k₂); o₂ } := by
      simp only [Option.mem_def, bind, Option.bind_eq_some, Option.guard_eq_some', exists_and_left,
        exists_const, and_imp]
      introv h h₁ h₂ h₃
      exact ⟨le_trans h₂ h, h₁ h₃⟩
    /-
      k k₂ : Nat
      c : Nat.Partrec.Code
      n x : Nat
      hl : LE.le (HAdd.hAdd k 1) (HAdd.hAdd k₂ 1)
      h : Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) c n) x
      hl' : LE.le k k₂
      this : ∀ {k k₂ n x : Nat} {o₁ o₂ : Option Nat}, LE.le k k₂ → (Membership.mem o …
      ⊢ Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k₂ 1) c n) x
    -/
    simp? at h ⊢ says simp only [Option.mem_def] at h ⊢
    /-
      k k₂ : Nat
      c : Nat.Partrec.Code
      n x : Nat
      hl : LE.le (HAdd.hAdd k 1) (HAdd.hAdd k₂ 1)
      hl' : LE.le k k₂
      this : ∀ {k k₂ n x : Nat} {o₁ o₂ : Option Nat}, LE.le k k₂ → (Membership.mem o …
      h : Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) c n) (Option.some x)
      ⊢ Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k₂ 1) c n) (Option.some x)
    -/
    induction' c with cf cg hf hg cf cg hf hg cf cg hf hg cf hf generalizing x n <;>
      /-
        case zero
        k k₂ : Nat
        hl : LE.le (HAdd.hAdd k 1) (HAdd.hAdd k₂ 1)
        hl' : LE.le k k₂
        this : ∀ {k k₂ n x : Nat} {o₁ o₂ : Option Nat}, LE.le k k₂ → (Membership.mem o …
        n x : Nat
        h : Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) Nat.Partrec.Code.zero n) (Optio …
        ⊢ Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k₂ 1) Nat.Partrec.Code.zero n) (Option …
      -/
      rw [evaln] at h ⊢ <;> refine this hl' (fun h => ?_) h
    /-
      case zero
      k k₂ : Nat
      hl : LE.le (HAdd.hAdd k 1) (HAdd.hAdd k₂ 1)
      hl' : LE.le k k₂
      this : ∀ {k k₂ n x : Nat} {o₁ o₂ : Option Nat}, LE.le k k₂ → (Membership.mem o …
      n x : Nat
      h✝ : Eq ((fun n => Bind.bind (guard (LE.le n k)) fun x => Pure.pure 0) n) (Opt …
      h : Membership.mem (Pure.pure 0) x
      ⊢ Membership.mem (Pure.pure 0) x
    -/
    iterate 4 exact h
    · -- pair cf cg
      simp? [Seq.seq, Option.bind_eq_some] at h ⊢ says
        simp only [Seq.seq, Option.map_eq_map, Option.mem_def, Option.bind_eq_some,
          Option.map_eq_some', exists_exists_and_eq_and] at h ⊢
      /-
        case pair
        k k₂ : Nat
        hl : LE.le (HAdd.hAdd k 1) (HAdd.hAdd k₂ 1)
        hl' : LE.le k k₂
        this : ∀ {k k₂ n x : Nat} {o₁ o₂ : Option Nat}, LE.le k k₂ → (Membership.mem o …
        cf cg : Nat.Partrec.Code
        hf : ∀ (n x : Nat), Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf n) (Option.s …
        hg : ∀ (n x : Nat), Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cg n) (Option.s …
        n x : Nat
        h✝ : Eq ((fun n => Bind.bind (guard (LE.le n k)) fun x => Seq.seq (Functor.map …
        h : Exists fun a => And (Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf n) (Opt …
        ⊢ Exists fun a => And (Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k₂ 1) cf n) (Opti …
      -/
      exact h.imp fun a => And.imp (hf _ _) <| Exists.imp fun b => And.imp_left (hg _ _)
      /-
        🎉 no goals
      -/
    · -- comp cf cg
      simp? [Bind.bind, Option.bind_eq_some] at h ⊢ says
        simp only [bind, Option.mem_def, Option.bind_eq_some] at h ⊢
      /-
        case comp
        k k₂ : Nat
        hl : LE.le (HAdd.hAdd k 1) (HAdd.hAdd k₂ 1)
        hl' : LE.le k k₂
        this : ∀ {k k₂ n x : Nat} {o₁ o₂ : Option Nat}, LE.le k k₂ → (Membership.mem o …
        cf cg : Nat.Partrec.Code
        hf : ∀ (n x : Nat), Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf n) (Option.s …
        hg : ∀ (n x : Nat), Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cg n) (Option.s …
        n x : Nat
        h✝ : Eq ((fun n => Bind.bind (guard (LE.le n k)) fun x => Bind.bind (Nat.Partr …
        h : Exists fun a => And (Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cg n) (Opt …
        ⊢ Exists fun a => And (Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k₂ 1) cg n) (Opti …
      -/
      exact h.imp fun a => And.imp (hg _ _) (hf _ _)
      /-
        🎉 no goals
      -/
    · -- prec cf cg
      /-
        case prec
        k k₂ : Nat
        hl : LE.le (HAdd.hAdd k 1) (HAdd.hAdd k₂ 1)
        hl' : LE.le k k₂
        this : ∀ {k k₂ n x : Nat} {o₁ o₂ : Option Nat}, LE.le k k₂ → (Membership.mem o …
        cf cg : Nat.Partrec.Code
        hf : ∀ (n x : Nat), Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf n) (Option.s …
        hg : ∀ (n x : Nat), Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cg n) (Option.s …
        n x : Nat
        h✝ : Eq ((fun n => Bind.bind (guard (LE.le n k)) fun x => Nat.unpaired (fun a  …
        h : Membership.mem (Nat.unpaired (fun a n => Nat.casesOn n (Nat.Partrec.Code.e …
        ⊢ Membership.mem (Nat.unpaired (fun a n => Nat.casesOn n (Nat.Partrec.Code.eva …
      -/
      revert h
      /-
        case prec
        k k₂ : Nat
        hl : LE.le (HAdd.hAdd k 1) (HAdd.hAdd k₂ 1)
        hl' : LE.le k k₂
        this : ∀ {k k₂ n x : Nat} {o₁ o₂ : Option Nat}, LE.le k k₂ → (Membership.mem o …
        cf cg : Nat.Partrec.Code
        hf : ∀ (n x : Nat), Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf n) (Option.s …
        hg : ∀ (n x : Nat), Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cg n) (Option.s …
        n x : Nat
        h : Eq ((fun n => Bind.bind (guard (LE.le n k)) fun x => Nat.unpaired (fun a n …
        ⊢ Membership.mem (Nat.unpaired (fun a n => Nat.casesOn n (Nat.Partrec.Code.eva …
      -/
      simp only [unpaired, bind, Option.mem_def]
      /-
        case prec
        k k₂ : Nat
        hl : LE.le (HAdd.hAdd k 1) (HAdd.hAdd k₂ 1)
        hl' : LE.le k k₂
        this : ∀ {k k₂ n x : Nat} {o₁ o₂ : Option Nat}, LE.le k k₂ → (Membership.mem o …
        cf cg : Nat.Partrec.Code
        hf : ∀ (n x : Nat), Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf n) (Option.s …
        hg : ∀ (n x : Nat), Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cg n) (Option.s …
        n x : Nat
        h : Eq ((fun n => Bind.bind (guard (LE.le n k)) fun x => Nat.unpaired (fun a n …
        ⊢ Eq (Nat.rec (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf (Nat.unpair n).1) (fu …
      -/
      induction n.unpair.2 <;> simp [Option.bind_eq_some]
        /-
          case prec.zero
          k k₂ : Nat
          hl : LE.le (HAdd.hAdd k 1) (HAdd.hAdd k₂ 1)
          hl' : LE.le k k₂
          this : ∀ {k k₂ n x : Nat} {o₁ o₂ : Option Nat}, LE.le k k₂ → (Membership.mem o …
          cf cg : Nat.Partrec.Code
          hf : ∀ (n x : Nat), Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf n) (Option.s …
          hg : ∀ (n x : Nat), Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cg n) (Option.s …
          n x : Nat
          h : Eq ((fun n => Bind.bind (guard (LE.le n k)) fun x => Nat.unpaired (fun a n …
          ⊢ Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf (Nat.unpair n).1) (Option.some …
        -/
      · apply hf
        /-
          🎉 no goals
        -/
        /-
          case prec.succ
          k k₂ : Nat
          hl : LE.le (HAdd.hAdd k 1) (HAdd.hAdd k₂ 1)
          hl' : LE.le k k₂
          this : ∀ {k k₂ n x : Nat} {o₁ o₂ : Option Nat}, LE.le k k₂ → (Membership.mem o …
          cf cg : Nat.Partrec.Code
          hf : ∀ (n x : Nat), Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf n) (Option.s …
          hg : ∀ (n x : Nat), Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cg n) (Option.s …
          n x : Nat
          h : Eq ((fun n => Bind.bind (guard (LE.le n k)) fun x => Nat.unpaired (fun a n …
          n✝ : Nat
          a✝ : Eq (Nat.rec (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf (Nat.unpair n).1)  …
          ⊢ ∀ (x_1 : Nat), Eq (Nat.Partrec.Code.evaln k (cf.prec cg) (Nat.pair (Nat.unpa …
        -/
      · exact fun y h₁ h₂ => ⟨y, evaln_mono hl' h₁, hg _ _ h₂⟩
        /-
          🎉 no goals
        -/
    · -- rfind' cf
      simp? [Bind.bind, Option.bind_eq_some] at h ⊢ says
        simp only [unpaired, bind, pair_unpair, Option.pure_def, Option.mem_def,
          Option.bind_eq_some] at h ⊢
      /-
        case rfind'
        k k₂ : Nat
        hl : LE.le (HAdd.hAdd k 1) (HAdd.hAdd k₂ 1)
        hl' : LE.le k k₂
        this : ∀ {k k₂ n x : Nat} {o₁ o₂ : Option Nat}, LE.le k k₂ → (Membership.mem o …
        cf : Nat.Partrec.Code
        hf : ∀ (n x : Nat), Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf n) (Option.s …
        n x : Nat
        h✝ : Eq ((fun n => Bind.bind (guard (LE.le n k)) fun x => Nat.unpaired (fun a  …
        h : Exists fun a => And (Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf n) (Opt …
        ⊢ Exists fun a => And (Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k₂ 1) cf n) (Opti …
      -/
      refine h.imp fun x => And.imp (hf _ _) ?_
      /-
        case rfind'
        k k₂ : Nat
        hl : LE.le (HAdd.hAdd k 1) (HAdd.hAdd k₂ 1)
        hl' : LE.le k k₂
        this : ∀ {k k₂ n x : Nat} {o₁ o₂ : Option Nat}, LE.le k k₂ → (Membership.mem o …
        cf : Nat.Partrec.Code
        hf : ∀ (n x : Nat), Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf n) (Option.s …
        n x✝ : Nat
        h✝ : Eq ((fun n => Bind.bind (guard (LE.le n k)) fun x => Nat.unpaired (fun a  …
        h : Exists fun a => And (Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf n) (Opt …
        x : Nat
        ⊢ Eq (ite (Eq x 0) (Option.some (Nat.unpair n).2) (Nat.Partrec.Code.evaln k cf …
      -/
                              /-
                                🎉 no goals
                              -/
      by_cases x0 : x = 0 <;> simp [x0]
      /-
        case neg
        k k₂ : Nat
        hl : LE.le (HAdd.hAdd k 1) (HAdd.hAdd k₂ 1)
        hl' : LE.le k k₂
        this : ∀ {k k₂ n x : Nat} {o₁ o₂ : Option Nat}, LE.le k k₂ → (Membership.mem o …
        cf : Nat.Partrec.Code
        hf : ∀ (n x : Nat), Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf n) (Option.s …
        n x✝ : Nat
        h✝ : Eq ((fun n => Bind.bind (guard (LE.le n k)) fun x => Nat.unpaired (fun a  …
        h : Exists fun a => And (Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf n) (Opt …
        x : Nat
        x0 : Not (Eq x 0)
        ⊢ Eq (Nat.Partrec.Code.evaln k cf.rfind' (Nat.pair (Nat.unpair n).1 (HAdd.hAdd …
      -/
      exact evaln_mono hl'
      /-
        🎉 no goals
      -/


theorem evaln_sound : ∀ {k c n x}, x ∈ evaln k c n → x ∈ eval c n
                        /-
                          x✝ : Nat.Partrec.Code
                          n x : Nat
                          h : Membership.mem (Nat.Partrec.Code.evaln 0 x✝ n) x
                          ⊢ Membership.mem (x✝.eval n) x
                        -/
  | 0, _, n, x, h => by simp [evaln] at h
                        /-
                          🎉 no goals
                        -/
  | k + 1, c, n, x, h => by
    /-
      k : Nat
      c : Nat.Partrec.Code
      n x : Nat
      h : Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) c n) x
      ⊢ Membership.mem (c.eval n) x
    -/
    induction' c with cf cg hf hg cf cg hf hg cf cg hf hg cf hf generalizing x n <;>
        /-
          case zero
          k n x : Nat
          h : Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) Nat.Partrec.Code.ze …
          ⊢ Membership.mem (Nat.Partrec.Code.zero.eval n) x
        -/
        simp [eval, evaln, Option.bind_eq_some, Seq.seq] at h ⊢ <;>
      /-
        case zero
        k n x : Nat
        h : And (LE.le n k) (Eq 0 x)
        ⊢ Membership.mem (Pure.pure 0 n) x
      -/
      cases' h with _ h
    /-
      case zero.intro
      k n x : Nat
      left✝ : LE.le n k
      h : Eq 0 x
      ⊢ Membership.mem (Pure.pure 0 n) x
    -/
    iterate 4 simpa [pure, PFun.pure, eq_comm] using h
    · -- pair cf cg
      /-
        case pair.intro
        k : Nat
        cf cg : Nat.Partrec.Code
        hf : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf  …
        hg : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cg  …
        n x : Nat
        left✝ : LE.le n k
        h : Exists fun a => And (Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf n) (Opt …
        ⊢ Exists fun a => And (Membership.mem (cf.eval n) a) (Exists fun a_1 => And (M …
      -/
      rcases h with ⟨y, ef, z, eg, rfl⟩
      /-
        case pair.intro.intro.intro.intro.intro
        k : Nat
        cf cg : Nat.Partrec.Code
        hf : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf  …
        hg : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cg  …
        n : Nat
        left✝ : LE.le n k
        y : Nat
        ef : Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf n) (Option.some y)
        z : Nat
        eg : Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cg n) (Option.some z)
        ⊢ Exists fun a => And (Membership.mem (cf.eval n) a) (Exists fun a_1 => And (M …
      -/
      exact ⟨_, hf _ _ ef, _, hg _ _ eg, rfl⟩
      /-
        🎉 no goals
      -/
    · --comp hf hg
      /-
        case comp.intro
        k : Nat
        cf cg : Nat.Partrec.Code
        hf : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf  …
        hg : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cg  …
        n x : Nat
        left✝ : LE.le n k
        h : Exists fun a => And (Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cg n) (Opt …
        ⊢ Exists fun a => And (Membership.mem (cg.eval n) a) (Membership.mem (cf.eval  …
      -/
      rcases h with ⟨y, eg, ef⟩
      /-
        case comp.intro.intro.intro
        k : Nat
        cf cg : Nat.Partrec.Code
        hf : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf  …
        hg : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cg  …
        n x : Nat
        left✝ : LE.le n k
        y : Nat
        eg : Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cg n) (Option.some y)
        ef : Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf y) (Option.some x)
        ⊢ Exists fun a => And (Membership.mem (cg.eval n) a) (Membership.mem (cf.eval  …
      -/
      exact ⟨_, hg _ _ eg, hf _ _ ef⟩
      /-
        🎉 no goals
      -/
    · -- prec cf cg
      /-
        case prec.intro
        k : Nat
        cf cg : Nat.Partrec.Code
        hf : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf  …
        hg : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cg  …
        n x : Nat
        left✝ : LE.le n k
        h : Eq (Nat.rec (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf (Nat.unpair n).1) ( …
        ⊢ Membership.mem (Nat.rec (cf.eval (Nat.unpair n).1) (fun y IH => IH.bind fun  …
      -/
      revert h
      /-
        case prec.intro
        k : Nat
        cf cg : Nat.Partrec.Code
        hf : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf  …
        hg : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cg  …
        n x : Nat
        left✝ : LE.le n k
        ⊢ Eq (Nat.rec (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf (Nat.unpair n).1) (fu …
      -/
      induction' n.unpair.2 with m IH generalizing x <;> simp [Option.bind_eq_some]
        /-
          case prec.intro.zero
          k : Nat
          cf cg : Nat.Partrec.Code
          hf : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf  …
          hg : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cg  …
          n : Nat
          left✝ : LE.le n k
          x : Nat
          ⊢ Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf (Nat.unpair n).1) (Option.some …
        -/
      · apply hf
        /-
          🎉 no goals
        -/
        /-
          case prec.intro.succ
          k : Nat
          cf cg : Nat.Partrec.Code
          hf : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf  …
          hg : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cg  …
          n : Nat
          left✝ : LE.le n k
          m : Nat
          IH : ∀ (x : Nat), Eq (Nat.rec (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf (Nat. …
          x : Nat
          ⊢ ∀ (x_1 : Nat), Eq (Nat.Partrec.Code.evaln k (cf.prec cg) (Nat.pair (Nat.unpa …
        -/
      · refine fun y h₁ h₂ => ⟨y, IH _ ?_, ?_⟩
          /-
            case prec.intro.succ.refine_1
            k : Nat
            cf cg : Nat.Partrec.Code
            hf : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf  …
            hg : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cg  …
            n : Nat
            left✝ : LE.le n k
            m : Nat
            IH : ∀ (x : Nat), Eq (Nat.rec (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf (Nat. …
            x y : Nat
            h₁ : Eq (Nat.Partrec.Code.evaln k (cf.prec cg) (Nat.pair (Nat.unpair n).1 m))  …
            h₂ : Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cg (Nat.pair (Nat.unpair n).1  …
            ⊢ Eq (Nat.rec (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf (Nat.unpair n).1) (fu …
          -/
        · have := evaln_mono k.le_succ h₁
          /-
            case prec.intro.succ.refine_1
            k : Nat
            cf cg : Nat.Partrec.Code
            hf : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf  …
            hg : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cg  …
            n : Nat
            left✝ : LE.le n k
            m : Nat
            IH : ∀ (x : Nat), Eq (Nat.rec (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf (Nat. …
            x y : Nat
            h₁ : Eq (Nat.Partrec.Code.evaln k (cf.prec cg) (Nat.pair (Nat.unpair n).1 m))  …
            h₂ : Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cg (Nat.pair (Nat.unpair n).1  …
            this : Membership.mem (Nat.Partrec.Code.evaln k.succ (cf.prec cg) (Nat.pair (N …
            ⊢ Eq (Nat.rec (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf (Nat.unpair n).1) (fu …
          -/
          simp [evaln, Option.bind_eq_some] at this
          /-
            case prec.intro.succ.refine_1
            k : Nat
            cf cg : Nat.Partrec.Code
            hf : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf  …
            hg : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cg  …
            n : Nat
            left✝ : LE.le n k
            m : Nat
            IH : ∀ (x : Nat), Eq (Nat.rec (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf (Nat. …
            x y : Nat
            h₁ : Eq (Nat.Partrec.Code.evaln k (cf.prec cg) (Nat.pair (Nat.unpair n).1 m))  …
            h₂ : Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cg (Nat.pair (Nat.unpair n).1  …
            this : And (LE.le (Nat.pair (Nat.unpair n).1 m) k) (Eq (Nat.rec (Nat.Partrec.C …
            ⊢ Eq (Nat.rec (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf (Nat.unpair n).1) (fu …
          -/
          exact this.2
          /-
            🎉 no goals
          -/
          /-
            case prec.intro.succ.refine_2
            k : Nat
            cf cg : Nat.Partrec.Code
            hf : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf  …
            hg : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cg  …
            n : Nat
            left✝ : LE.le n k
            m : Nat
            IH : ∀ (x : Nat), Eq (Nat.rec (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf (Nat. …
            x y : Nat
            h₁ : Eq (Nat.Partrec.Code.evaln k (cf.prec cg) (Nat.pair (Nat.unpair n).1 m))  …
            h₂ : Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cg (Nat.pair (Nat.unpair n).1  …
            ⊢ Membership.mem (cg.eval (Nat.pair (Nat.unpair n).1 (Nat.pair m y))) x
          -/
        · exact hg _ _ h₂
          /-
            🎉 no goals
          -/
    · -- rfind' cf
      /-
        case rfind'.intro
        k : Nat
        cf : Nat.Partrec.Code
        hf : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf  …
        n x : Nat
        left✝ : LE.le n k
        h : Exists fun a => And (Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf n) (Opt …
        ⊢ Exists fun a => And (And (Membership.mem (cf.eval (Nat.pair (Nat.unpair n).1 …
      -/
      rcases h with ⟨m, h₁, h₂⟩
      /-
        case rfind'.intro.intro.intro
        k : Nat
        cf : Nat.Partrec.Code
        hf : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf  …
        n x : Nat
        left✝ : LE.le n k
        m : Nat
        h₁ : Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf n) (Option.some m)
        h₂ : Eq (ite (Eq m 0) (Option.some (Nat.unpair n).2) (Nat.Partrec.Code.evaln k …
        ⊢ Exists fun a => And (And (Membership.mem (cf.eval (Nat.pair (Nat.unpair n).1 …
      -/
      by_cases m0 : m = 0 <;> simp [m0] at h₂
      · exact
          ⟨0, ⟨by simpa [m0] using hf _ _ h₁, fun {m} => (Nat.not_lt_zero _).elim⟩, by simp [h₂]⟩
        /-
          case neg
          k : Nat
          cf : Nat.Partrec.Code
          hf : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf  …
          n x : Nat
          left✝ : LE.le n k
          m : Nat
          h₁ : Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf n) (Option.some m)
          m0 : Not (Eq m 0)
          h₂ : Eq (Nat.Partrec.Code.evaln k cf.rfind' (Nat.pair (Nat.unpair n).1 (HAdd.h …
          ⊢ Exists fun a => And (And (Membership.mem (cf.eval (Nat.pair (Nat.unpair n).1 …
        -/
      · have := evaln_sound h₂
        /-
          case neg
          k : Nat
          cf : Nat.Partrec.Code
          hf : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf  …
          n x : Nat
          left✝ : LE.le n k
          m : Nat
          h₁ : Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf n) (Option.some m)
          m0 : Not (Eq m 0)
          h₂ : Eq (Nat.Partrec.Code.evaln k cf.rfind' (Nat.pair (Nat.unpair n).1 (HAdd.h …
          this : Membership.mem (cf.rfind'.eval (Nat.pair (Nat.unpair n).1 (HAdd.hAdd (N …
          ⊢ Exists fun a => And (And (Membership.mem (cf.eval (Nat.pair (Nat.unpair n).1 …
        -/
        simp [eval] at this
        /-
          case neg
          k : Nat
          cf : Nat.Partrec.Code
          hf : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf  …
          n x : Nat
          left✝ : LE.le n k
          m : Nat
          h₁ : Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf n) (Option.some m)
          m0 : Not (Eq m 0)
          h₂ : Eq (Nat.Partrec.Code.evaln k cf.rfind' (Nat.pair (Nat.unpair n).1 (HAdd.h …
          this : Exists fun a => And (And (Membership.mem (cf.eval (Nat.pair (Nat.unpair …
          ⊢ Exists fun a => And (And (Membership.mem (cf.eval (Nat.pair (Nat.unpair n).1 …
        -/
        rcases this with ⟨y, ⟨hy₁, hy₂⟩, rfl⟩
        refine
          ⟨y + 1, ⟨by simpa [add_comm, add_left_comm] using hy₁, fun {i} im => ?_⟩, by
            simp [add_comm, add_left_comm]⟩
        /-
          case neg.intro.intro.intro
          k : Nat
          cf : Nat.Partrec.Code
          hf : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf  …
          n : Nat
          left✝ : LE.le n k
          m : Nat
          h₁ : Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf n) (Option.some m)
          m0 : Not (Eq m 0)
          y : Nat
          hy₁ : Membership.mem (cf.eval (Nat.pair (Nat.unpair n).1 (HAdd.hAdd y (HAdd.hA …
          hy₂ : ∀ {m : Nat}, LT.lt m y → Exists fun a => And (Membership.mem (cf.eval (N …
          h₂ : Eq (Nat.Partrec.Code.evaln k cf.rfind' (Nat.pair (Nat.unpair n).1 (HAdd.h …
          i : Nat
          im : LT.lt i (HAdd.hAdd y 1)
          ⊢ Exists fun a => And (Membership.mem (cf.eval (Nat.pair (Nat.unpair n).1 (HAd …
        -/
        cases' i with i
          /-
            case neg.intro.intro.intro.zero
            k : Nat
            cf : Nat.Partrec.Code
            hf : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf  …
            n : Nat
            left✝ : LE.le n k
            m : Nat
            h₁ : Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf n) (Option.some m)
            m0 : Not (Eq m 0)
            y : Nat
            hy₁ : Membership.mem (cf.eval (Nat.pair (Nat.unpair n).1 (HAdd.hAdd y (HAdd.hA …
            hy₂ : ∀ {m : Nat}, LT.lt m y → Exists fun a => And (Membership.mem (cf.eval (N …
            h₂ : Eq (Nat.Partrec.Code.evaln k cf.rfind' (Nat.pair (Nat.unpair n).1 (HAdd.h …
            im : LT.lt 0 (HAdd.hAdd y 1)
            ⊢ Exists fun a => And (Membership.mem (cf.eval (Nat.pair (Nat.unpair n).1 (HAd …
          -/
        · exact ⟨m, by simpa using hf _ _ h₁, m0⟩
          /-
            🎉 no goals
          -/
          /-
            case neg.intro.intro.intro.succ
            k : Nat
            cf : Nat.Partrec.Code
            hf : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf  …
            n : Nat
            left✝ : LE.le n k
            m : Nat
            h₁ : Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf n) (Option.some m)
            m0 : Not (Eq m 0)
            y : Nat
            hy₁ : Membership.mem (cf.eval (Nat.pair (Nat.unpair n).1 (HAdd.hAdd y (HAdd.hA …
            hy₂ : ∀ {m : Nat}, LT.lt m y → Exists fun a => And (Membership.mem (cf.eval (N …
            h₂ : Eq (Nat.Partrec.Code.evaln k cf.rfind' (Nat.pair (Nat.unpair n).1 (HAdd.h …
            i : Nat
            im : LT.lt (HAdd.hAdd i 1) (HAdd.hAdd y 1)
            ⊢ Exists fun a => And (Membership.mem (cf.eval (Nat.pair (Nat.unpair n).1 (HAd …
          -/
        · rcases hy₂ (Nat.lt_of_succ_lt_succ im) with ⟨z, hz, z0⟩
          /-
            case neg.intro.intro.intro.succ.intro.intro
            k : Nat
            cf : Nat.Partrec.Code
            hf : ∀ (n x : Nat), Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf  …
            n : Nat
            left✝ : LE.le n k
            m : Nat
            h₁ : Eq (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) cf n) (Option.some m)
            m0 : Not (Eq m 0)
            y : Nat
            hy₁ : Membership.mem (cf.eval (Nat.pair (Nat.unpair n).1 (HAdd.hAdd y (HAdd.hA …
            hy₂ : ∀ {m : Nat}, LT.lt m y → Exists fun a => And (Membership.mem (cf.eval (N …
            h₂ : Eq (Nat.Partrec.Code.evaln k cf.rfind' (Nat.pair (Nat.unpair n).1 (HAdd.h …
            i : Nat
            im : LT.lt (HAdd.hAdd i 1) (HAdd.hAdd y 1)
            z : Nat
            hz : Membership.mem (cf.eval (Nat.pair (Nat.unpair n).1 (HAdd.hAdd i (HAdd.hAd …
            z0 : Not (Eq z 0)
            ⊢ Exists fun a => And (Membership.mem (cf.eval (Nat.pair (Nat.unpair n).1 (HAd …
          -/
          exact ⟨z, by simpa [add_comm, add_left_comm] using hz, z0⟩
          /-
            🎉 no goals
          -/


theorem evaln_complete {c n x} : x ∈ eval c n ↔ ∃ k, x ∈ evaln k c n := by
  /-
    c : Nat.Partrec.Code
    n x : Nat
    ⊢ Iff (Membership.mem (c.eval n) x) (Exists fun k => Membership.mem (Nat.Partr …
  -/
  refine ⟨fun h => ?_, fun ⟨k, h⟩ => evaln_sound h⟩
  /-
    c : Nat.Partrec.Code
    n x : Nat
    h : Membership.mem (c.eval n) x
    ⊢ Exists fun k => Membership.mem (Nat.Partrec.Code.evaln k c n) x
  -/
  rsuffices ⟨k, h⟩ : ∃ k, x ∈ evaln (k + 1) c n
    /-
      case intro
      c : Nat.Partrec.Code
      n x : Nat
      h✝ : Membership.mem (c.eval n) x
      k : Nat
      h : Membership.mem (Nat.Partrec.Code.evaln (HAdd.hAdd k 1) c n) x
      ⊢ Exists fun k => Membership.mem (Nat.Partrec.Code.evaln k c n) x
    -/
  · exact ⟨k + 1, h⟩
    /-
      🎉 no goals
    -/
  induction c generalizing n x with
      simp [eval, evaln, pure, PFun.pure, Seq.seq, Option.bind_eq_some] at h ⊢
  | pair cf cg hf hg =>
    rcases h with ⟨x, hx, y, hy, rfl⟩
    rcases hf hx with ⟨k₁, hk₁⟩; rcases hg hy with ⟨k₂, hk₂⟩
    refine ⟨max k₁ k₂, ?_⟩
    refine
      ⟨le_max_of_le_left <| Nat.le_of_lt_succ <| evaln_bound hk₁, _,
        evaln_mono (Nat.succ_le_succ <| le_max_left _ _) hk₁, _,
        evaln_mono (Nat.succ_le_succ <| le_max_right _ _) hk₂, rfl⟩
  | comp cf cg hf hg =>
    rcases h with ⟨y, hy, hx⟩
    rcases hg hy with ⟨k₁, hk₁⟩; rcases hf hx with ⟨k₂, hk₂⟩
    refine ⟨max k₁ k₂, ?_⟩
    exact
      ⟨le_max_of_le_left <| Nat.le_of_lt_succ <| evaln_bound hk₁, _,
        evaln_mono (Nat.succ_le_succ <| le_max_left _ _) hk₁,
        evaln_mono (Nat.succ_le_succ <| le_max_right _ _) hk₂⟩
  | prec cf cg hf hg =>
    revert h
    generalize n.unpair.1 = n₁; generalize n.unpair.2 = n₂
    induction' n₂ with m IH generalizing x n <;> simp [Option.bind_eq_some]
    · intro h
      rcases hf h with ⟨k, hk⟩
      exact ⟨_, le_max_left _ _, evaln_mono (Nat.succ_le_succ <| le_max_right _ _) hk⟩
    · intro y hy hx
      rcases IH hy with ⟨k₁, nk₁, hk₁⟩
      rcases hg hx with ⟨k₂, hk₂⟩
      refine
        ⟨(max k₁ k₂).succ,
          Nat.le_succ_of_le <| le_max_of_le_left <|
            le_trans (le_max_left _ (Nat.pair n₁ m)) nk₁, y,
          evaln_mono (Nat.succ_le_succ <| le_max_left _ _) ?_,
          evaln_mono (Nat.succ_le_succ <| Nat.le_succ_of_le <| le_max_right _ _) hk₂⟩
      simp only [evaln.eq_8, bind, unpaired, unpair_pair, Option.mem_def, Option.bind_eq_some,
        Option.guard_eq_some', exists_and_left, exists_const]
      exact ⟨le_trans (le_max_right _ _) nk₁, hk₁⟩
  | rfind' cf hf =>
    rcases h with ⟨y, ⟨hy₁, hy₂⟩, rfl⟩
    suffices ∃ k, y + n.unpair.2 ∈ evaln (k + 1) (rfind' cf) (Nat.pair n.unpair.1 n.unpair.2) by
      simpa [evaln, Option.bind_eq_some]
    revert hy₁ hy₂
    generalize n.unpair.2 = m
    intro hy₁ hy₂
    induction' y with y IH generalizing m <;> simp [evaln, Option.bind_eq_some]
    · simp at hy₁
      rcases hf hy₁ with ⟨k, hk⟩
      exact ⟨_, Nat.le_of_lt_succ <| evaln_bound hk, _, hk, by simp⟩
    · rcases hy₂ (Nat.succ_pos _) with ⟨a, ha, a0⟩
      rcases hf ha with ⟨k₁, hk₁⟩
      rcases IH m.succ (by simpa [Nat.succ_eq_add_one, add_comm, add_left_comm] using hy₁)
          fun {i} hi => by
          simpa [Nat.succ_eq_add_one, add_comm, add_left_comm] using
            hy₂ (Nat.succ_lt_succ hi) with
        ⟨k₂, hk₂⟩
      use (max k₁ k₂).succ
      rw [zero_add] at hk₁
      use Nat.le_succ_of_le <| le_max_of_le_left <| Nat.le_of_lt_succ <| evaln_bound hk₁
      use a
      use evaln_mono (Nat.succ_le_succ <| Nat.le_succ_of_le <| le_max_left _ _) hk₁
      simpa [a0, add_comm, add_left_comm] using
        evaln_mono (Nat.succ_le_succ <| le_max_right _ _) hk₂
  | _ => exact ⟨⟨_, le_rfl⟩, h.symm⟩


private def lup (L : List (List (Option ℕ))) (p : ℕ × Code) (n : ℕ) := do
  let l ← L.get? (encode p)
  let o ← l.get? n
  o


private theorem hlup : Primrec fun p : _ × (_ × _) × _ => lup p.1 p.2.1 p.2.2 :=
  Primrec.option_bind
    (Primrec.list_get?.comp Primrec.fst (Primrec.encode.comp <| Primrec.fst.comp Primrec.snd))
    (Primrec.option_bind (Primrec.list_get?.comp Primrec.snd <| Primrec.snd.comp <|
      Primrec.snd.comp Primrec.fst) Primrec.snd)


private def G (L : List (List (Option ℕ))) : Option (List (Option ℕ)) :=
  Option.some <|
    let a := ofNat (ℕ × Code) L.length
    let k := a.1
    let c := a.2
    (List.range k).map fun n =>
      k.casesOn Option.none fun k' =>
        Nat.Partrec.Code.recOn c
          (some 0) -- zero
          (some (Nat.succ n))
          (some n.unpair.1)
          (some n.unpair.2)
          (fun cf cg _ _ => do
            let x ← lup L (k, cf) n
            let y ← lup L (k, cg) n
            some (Nat.pair x y))
          (fun cf cg _ _ => do
            let x ← lup L (k, cg) n
            lup L (k, cf) x)
          (fun cf cg _ _ =>
            let z := n.unpair.1
            n.unpair.2.casesOn (lup L (k, cf) z) fun y => do
              let i ← lup L (k', c) (Nat.pair z y)
              lup L (k, cg) (Nat.pair z (Nat.pair y i)))
          (fun cf _ =>
            let z := n.unpair.1
            let m := n.unpair.2
            do
              let x ← lup L (k, cf) (Nat.pair z m)
              x.casesOn (some m) fun _ => lup L (k', c) (Nat.pair z (m + 1)))


private theorem hG : Primrec G := by
  /-
    ⊢ Primrec Nat.Partrec.Code.G
  -/
  have a := (Primrec.ofNat (ℕ × Code)).comp (Primrec.list_length (α := List (Option ℕ)))
  /-
    a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
    ⊢ Primrec Nat.Partrec.Code.G
  -/
  have k := Primrec.fst.comp a
  /-
    a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
    k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length).1
    ⊢ Primrec Nat.Partrec.Code.G
  -/
  refine Primrec.option_some.comp (Primrec.list_map (Primrec.list_range.comp k) (?_ : Primrec _))
  /-
    a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
    k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length).1
    ⊢ Primrec fun p =>
        (fun a n =>
            Nat.casesOn (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length).1 …
              Nat.Partrec.Code.recOn (Denumerable.ofNat (Prod Nat Nat.Partrec.Code …
                (fun cf cg x x =>
                  let z := (Nat.unpair n).1;
                  Nat.casesOn (Nat.unpair n).2 (Nat.Partrec.Code.lup a { fst := (D …
                fun cf x =>
                let z := (Nat.unpair n).1;
                let m := (Nat.unpair n).2;
                Bind.bind (Nat.Partrec.Code.lup a { fst := (Denumerable.ofNat (Pro …
          p.1 p.2
  -/
  replace k := k.comp (Primrec.fst (β := ℕ))
  /-
    a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
    k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.length …
    ⊢ Primrec fun p =>
        (fun a n =>
            Nat.casesOn (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length).1 …
              Nat.Partrec.Code.recOn (Denumerable.ofNat (Prod Nat Nat.Partrec.Code …
                (fun cf cg x x =>
                  let z := (Nat.unpair n).1;
                  Nat.casesOn (Nat.unpair n).2 (Nat.Partrec.Code.lup a { fst := (D …
                fun cf x =>
                let z := (Nat.unpair n).1;
                let m := (Nat.unpair n).2;
                Bind.bind (Nat.Partrec.Code.lup a { fst := (Denumerable.ofNat (Pro …
          p.1 p.2
  -/
  have n := Primrec.snd (α := List (List (Option ℕ))) (β := ℕ)
  /-
    a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
    k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.length …
    n : Primrec Prod.snd
    ⊢ Primrec fun p =>
        (fun a n =>
            Nat.casesOn (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length).1 …
              Nat.Partrec.Code.recOn (Denumerable.ofNat (Prod Nat Nat.Partrec.Code …
                (fun cf cg x x =>
                  let z := (Nat.unpair n).1;
                  Nat.casesOn (Nat.unpair n).2 (Nat.Partrec.Code.lup a { fst := (D …
                fun cf x =>
                let z := (Nat.unpair n).1;
                let m := (Nat.unpair n).2;
                Bind.bind (Nat.Partrec.Code.lup a { fst := (Denumerable.ofNat (Pro …
          p.1 p.2
  -/
  refine Primrec.nat_casesOn k (_root_.Primrec.const Option.none) (?_ : Primrec _)
  /-
    a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
    k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.length …
    n : Primrec Prod.snd
    ⊢ Primrec fun p =>
        (fun p k' =>
            Nat.Partrec.Code.recOn (Denumerable.ofNat (Prod Nat Nat.Partrec.Code)  …
              (fun cf cg x x =>
                let z := (Nat.unpair p.2).1;
                Nat.casesOn (Nat.unpair p.2).2 (Nat.Partrec.Code.lup p.1 { fst :=  …
              fun cf x =>
              let z := (Nat.unpair p.2).1;
              let m := (Nat.unpair p.2).2;
              Bind.bind (Nat.Partrec.Code.lup p.1 { fst := (Denumerable.ofNat (Pro …
          p.1 p.2
  -/
  have k := k.comp (Primrec.fst (β := ℕ))
  /-
    a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
    k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.lengt …
    n : Primrec Prod.snd
    k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
    ⊢ Primrec fun p =>
        (fun p k' =>
            Nat.Partrec.Code.recOn (Denumerable.ofNat (Prod Nat Nat.Partrec.Code)  …
              (fun cf cg x x =>
                let z := (Nat.unpair p.2).1;
                Nat.casesOn (Nat.unpair p.2).2 (Nat.Partrec.Code.lup p.1 { fst :=  …
              fun cf x =>
              let z := (Nat.unpair p.2).1;
              let m := (Nat.unpair p.2).2;
              Bind.bind (Nat.Partrec.Code.lup p.1 { fst := (Denumerable.ofNat (Pro …
          p.1 p.2
  -/
  have n := n.comp (Primrec.fst (β := ℕ))
  /-
    a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
    k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.lengt …
    n✝ : Primrec Prod.snd
    k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
    n : Primrec fun a => a.1.2
    ⊢ Primrec fun p =>
        (fun p k' =>
            Nat.Partrec.Code.recOn (Denumerable.ofNat (Prod Nat Nat.Partrec.Code)  …
              (fun cf cg x x =>
                let z := (Nat.unpair p.2).1;
                Nat.casesOn (Nat.unpair p.2).2 (Nat.Partrec.Code.lup p.1 { fst :=  …
              fun cf x =>
              let z := (Nat.unpair p.2).1;
              let m := (Nat.unpair p.2).2;
              Bind.bind (Nat.Partrec.Code.lup p.1 { fst := (Denumerable.ofNat (Pro …
          p.1 p.2
  -/
  have k' := Primrec.snd (α := List (List (Option ℕ)) × ℕ) (β := ℕ)
  /-
    a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
    k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.lengt …
    n✝ : Primrec Prod.snd
    k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
    n : Primrec fun a => a.1.2
    k' : Primrec Prod.snd
    ⊢ Primrec fun p =>
        (fun p k' =>
            Nat.Partrec.Code.recOn (Denumerable.ofNat (Prod Nat Nat.Partrec.Code)  …
              (fun cf cg x x =>
                let z := (Nat.unpair p.2).1;
                Nat.casesOn (Nat.unpair p.2).2 (Nat.Partrec.Code.lup p.1 { fst :=  …
              fun cf x =>
              let z := (Nat.unpair p.2).1;
              let m := (Nat.unpair p.2).2;
              Bind.bind (Nat.Partrec.Code.lup p.1 { fst := (Denumerable.ofNat (Pro …
          p.1 p.2
  -/
  have c := Primrec.snd.comp (a.comp <| (Primrec.fst (β := ℕ)).comp (Primrec.fst (β := ℕ)))
  apply
    Nat.Partrec.Code.rec_prim c
      (_root_.Primrec.const (some 0))
      (Primrec.option_some.comp (_root_.Primrec.succ.comp n))
      (Primrec.option_some.comp (Primrec.fst.comp <| Primrec.unpair.comp n))
      (Primrec.option_some.comp (Primrec.snd.comp <| Primrec.unpair.comp n))
  · have L := (Primrec.fst.comp Primrec.fst).comp
      (Primrec.fst (α := (List (List (Option ℕ)) × ℕ) × ℕ)
        (β := Code × Code × Option ℕ × Option ℕ))
    /-
      case hpr
      a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
      k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.lengt …
      n✝ : Primrec Prod.snd
      k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      n : Primrec fun a => a.1.2
      k' : Primrec Prod.snd
      c : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      L : Primrec fun a => a.1.1.1
      ⊢ Primrec fun a => Bind.bind (Nat.Partrec.Code.lup a.1.1.1 { fst := (Denumerab …
    -/
    have k := k.comp (Primrec.fst (β := Code × Code × Option ℕ × Option ℕ))
    /-
      case hpr
      a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
      k✝¹ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.leng …
      n✝ : Primrec Prod.snd
      k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.len …
      n : Primrec fun a => a.1.2
      k' : Primrec Prod.snd
      c : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      L : Primrec fun a => a.1.1.1
      k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.1.le …
      ⊢ Primrec fun a => Bind.bind (Nat.Partrec.Code.lup a.1.1.1 { fst := (Denumerab …
    -/
    have n := n.comp (Primrec.fst (β := Code × Code × Option ℕ × Option ℕ))
    have cf := Primrec.fst.comp (Primrec.snd (α := (List (List (Option ℕ)) × ℕ) × ℕ)
        (β := Code × Code × Option ℕ × Option ℕ))
    have cg := (Primrec.fst.comp Primrec.snd).comp
      (Primrec.snd (α := (List (List (Option ℕ)) × ℕ) × ℕ)
        (β := Code × Code × Option ℕ × Option ℕ))
    /-
      case hpr
      a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
      k✝¹ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.leng …
      n✝¹ : Primrec Prod.snd
      k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.len …
      n✝ : Primrec fun a => a.1.2
      k' : Primrec Prod.snd
      c : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      L : Primrec fun a => a.1.1.1
      k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.1.le …
      n : Primrec fun a => a.1.1.2
      cf : Primrec fun a => a.2.1
      cg : Primrec fun a => a.2.2.1
      ⊢ Primrec fun a => Bind.bind (Nat.Partrec.Code.lup a.1.1.1 { fst := (Denumerab …
    -/
    refine Primrec.option_bind (hlup.comp <| L.pair <| (k.pair cf).pair n) ?_
    /-
      case hpr
      a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
      k✝¹ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.leng …
      n✝¹ : Primrec Prod.snd
      k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.len …
      n✝ : Primrec fun a => a.1.2
      k' : Primrec Prod.snd
      c : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      L : Primrec fun a => a.1.1.1
      k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.1.le …
      n : Primrec fun a => a.1.1.2
      cf : Primrec fun a => a.2.1
      cg : Primrec fun a => a.2.2.1
      ⊢ Primrec₂ fun a x => Bind.bind (Nat.Partrec.Code.lup a.1.1.1 { fst := (Denume …
    -/
    unfold Primrec₂
    conv =>
      congr
      · ext p
        dsimp only []
        erw [Option.bind_eq_bind, ← Option.map_eq_bind]
    /-
      case hpr
      a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
      k✝¹ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.leng …
      n✝¹ : Primrec Prod.snd
      k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.len …
      n✝ : Primrec fun a => a.1.2
      k' : Primrec Prod.snd
      c : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      L : Primrec fun a => a.1.1.1
      k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.1.le …
      n : Primrec fun a => a.1.1.2
      cf : Primrec fun a => a.2.1
      cg : Primrec fun a => a.2.2.1
      ⊢ Primrec fun p => Option.map (Nat.pair p.2) (Nat.Partrec.Code.lup p.1.1.1.1 { …
    -/
    refine Primrec.option_map ((hlup.comp <| L.pair <| (k.pair cg).pair n).comp Primrec.fst) ?_
    /-
      case hpr
      a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
      k✝¹ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.leng …
      n✝¹ : Primrec Prod.snd
      k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.len …
      n✝ : Primrec fun a => a.1.2
      k' : Primrec Prod.snd
      c : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      L : Primrec fun a => a.1.1.1
      k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.1.le …
      n : Primrec fun a => a.1.1.2
      cf : Primrec fun a => a.2.1
      cg : Primrec fun a => a.2.2.1
      ⊢ Primrec₂ fun p => Nat.pair p.2
    -/
    unfold Primrec₂
    /-
      case hpr
      a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
      k✝¹ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.leng …
      n✝¹ : Primrec Prod.snd
      k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.len …
      n✝ : Primrec fun a => a.1.2
      k' : Primrec Prod.snd
      c : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      L : Primrec fun a => a.1.1.1
      k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.1.le …
      n : Primrec fun a => a.1.1.2
      cf : Primrec fun a => a.2.1
      cg : Primrec fun a => a.2.2.1
      ⊢ Primrec fun p => (fun p => Nat.pair p.2) p.1 p.2
    -/
    exact Primrec₂.natPair.comp (Primrec.snd.comp Primrec.fst) Primrec.snd
    /-
      🎉 no goals
    -/
  · have L := (Primrec.fst.comp Primrec.fst).comp
      (Primrec.fst (α := (List (List (Option ℕ)) × ℕ) × ℕ)
        (β := Code × Code × Option ℕ × Option ℕ))
    /-
      case hco
      a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
      k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.lengt …
      n✝ : Primrec Prod.snd
      k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      n : Primrec fun a => a.1.2
      k' : Primrec Prod.snd
      c : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      L : Primrec fun a => a.1.1.1
      ⊢ Primrec fun a => Bind.bind (Nat.Partrec.Code.lup a.1.1.1 { fst := (Denumerab …
    -/
    have k := k.comp (Primrec.fst (β := Code × Code × Option ℕ × Option ℕ))
    /-
      case hco
      a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
      k✝¹ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.leng …
      n✝ : Primrec Prod.snd
      k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.len …
      n : Primrec fun a => a.1.2
      k' : Primrec Prod.snd
      c : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      L : Primrec fun a => a.1.1.1
      k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.1.le …
      ⊢ Primrec fun a => Bind.bind (Nat.Partrec.Code.lup a.1.1.1 { fst := (Denumerab …
    -/
    have n := n.comp (Primrec.fst (β := Code × Code × Option ℕ × Option ℕ))
    have cf := Primrec.fst.comp (Primrec.snd (α := (List (List (Option ℕ)) × ℕ) × ℕ)
        (β := Code × Code × Option ℕ × Option ℕ))
    have cg := (Primrec.fst.comp Primrec.snd).comp
      (Primrec.snd (α := (List (List (Option ℕ)) × ℕ) × ℕ)
        (β := Code × Code × Option ℕ × Option ℕ))
    /-
      case hco
      a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
      k✝¹ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.leng …
      n✝¹ : Primrec Prod.snd
      k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.len …
      n✝ : Primrec fun a => a.1.2
      k' : Primrec Prod.snd
      c : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      L : Primrec fun a => a.1.1.1
      k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.1.le …
      n : Primrec fun a => a.1.1.2
      cf : Primrec fun a => a.2.1
      cg : Primrec fun a => a.2.2.1
      ⊢ Primrec fun a => Bind.bind (Nat.Partrec.Code.lup a.1.1.1 { fst := (Denumerab …
    -/
    refine Primrec.option_bind (hlup.comp <| L.pair <| (k.pair cg).pair n) ?_
    /-
      case hco
      a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
      k✝¹ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.leng …
      n✝¹ : Primrec Prod.snd
      k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.len …
      n✝ : Primrec fun a => a.1.2
      k' : Primrec Prod.snd
      c : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      L : Primrec fun a => a.1.1.1
      k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.1.le …
      n : Primrec fun a => a.1.1.2
      cf : Primrec fun a => a.2.1
      cg : Primrec fun a => a.2.2.1
      ⊢ Primrec₂ fun a x => Nat.Partrec.Code.lup a.1.1.1 { fst := (Denumerable.ofNat …
    -/
    unfold Primrec₂
    have h :=
      hlup.comp ((L.comp Primrec.fst).pair <| ((k.pair cf).comp Primrec.fst).pair Primrec.snd)
    /-
      case hco
      a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
      k✝¹ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.leng …
      n✝¹ : Primrec Prod.snd
      k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.len …
      n✝ : Primrec fun a => a.1.2
      k' : Primrec Prod.snd
      c : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      L : Primrec fun a => a.1.1.1
      k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.1.le …
      n : Primrec fun a => a.1.1.2
      cf : Primrec fun a => a.2.1
      cg : Primrec fun a => a.2.2.1
      h : Primrec fun a => Nat.Partrec.Code.lup { fst := a.1.1.1.1, snd := { fst :=  …
      ⊢ Primrec fun p => (fun a x => Nat.Partrec.Code.lup a.1.1.1 { fst := (Denumera …
    -/
    exact h
    /-
      🎉 no goals
    -/
  · have L := (Primrec.fst.comp Primrec.fst).comp
      (Primrec.fst (α := (List (List (Option ℕ)) × ℕ) × ℕ)
        (β := Code × Code × Option ℕ × Option ℕ))
    /-
      case hpc
      a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
      k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.lengt …
      n✝ : Primrec Prod.snd
      k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      n : Primrec fun a => a.1.2
      k' : Primrec Prod.snd
      c : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      L : Primrec fun a => a.1.1.1
      ⊢ Primrec fun a =>
          let z := (Nat.unpair a.1.1.2).1;
          Nat.casesOn (Nat.unpair a.1.1.2).2 (Nat.Partrec.Code.lup a.1.1.1 { fst :=  …
    -/
    have k := k.comp (Primrec.fst (β := Code × Code × Option ℕ × Option ℕ))
    /-
      case hpc
      a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
      k✝¹ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.leng …
      n✝ : Primrec Prod.snd
      k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.len …
      n : Primrec fun a => a.1.2
      k' : Primrec Prod.snd
      c : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      L : Primrec fun a => a.1.1.1
      k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.1.le …
      ⊢ Primrec fun a =>
          let z := (Nat.unpair a.1.1.2).1;
          Nat.casesOn (Nat.unpair a.1.1.2).2 (Nat.Partrec.Code.lup a.1.1.1 { fst :=  …
    -/
    have n := n.comp (Primrec.fst (β := Code × Code × Option ℕ × Option ℕ))
    have cf := Primrec.fst.comp (Primrec.snd (α := (List (List (Option ℕ)) × ℕ) × ℕ)
        (β := Code × Code × Option ℕ × Option ℕ))
    have cg := (Primrec.fst.comp Primrec.snd).comp
      (Primrec.snd (α := (List (List (Option ℕ)) × ℕ) × ℕ)
        (β := Code × Code × Option ℕ × Option ℕ))
    /-
      case hpc
      a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
      k✝¹ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.leng …
      n✝¹ : Primrec Prod.snd
      k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.len …
      n✝ : Primrec fun a => a.1.2
      k' : Primrec Prod.snd
      c : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      L : Primrec fun a => a.1.1.1
      k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.1.le …
      n : Primrec fun a => a.1.1.2
      cf : Primrec fun a => a.2.1
      cg : Primrec fun a => a.2.2.1
      ⊢ Primrec fun a =>
          let z := (Nat.unpair a.1.1.2).1;
          Nat.casesOn (Nat.unpair a.1.1.2).2 (Nat.Partrec.Code.lup a.1.1.1 { fst :=  …
    -/
    have z := Primrec.fst.comp (Primrec.unpair.comp n)
    refine
      Primrec.nat_casesOn (Primrec.snd.comp (Primrec.unpair.comp n))
        (hlup.comp <| L.pair <| (k.pair cf).pair z)
        (?_ : Primrec _)
    /-
      case hpc
      a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
      k✝¹ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.leng …
      n✝¹ : Primrec Prod.snd
      k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.len …
      n✝ : Primrec fun a => a.1.2
      k' : Primrec Prod.snd
      c : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      L : Primrec fun a => a.1.1.1
      k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.1.le …
      n : Primrec fun a => a.1.1.2
      cf : Primrec fun a => a.2.1
      cg : Primrec fun a => a.2.2.1
      z : Primrec fun a => (Nat.unpair a.1.1.2).1
      ⊢ Primrec fun p => (fun a y => Bind.bind (Nat.Partrec.Code.lup a.1.1.1 { fst : …
    -/
    have L := L.comp (Primrec.fst (β := ℕ))
    /-
      case hpc
      a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
      k✝¹ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.leng …
      n✝¹ : Primrec Prod.snd
      k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.len …
      n✝ : Primrec fun a => a.1.2
      k' : Primrec Prod.snd
      c : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      L✝ : Primrec fun a => a.1.1.1
      k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.1.le …
      n : Primrec fun a => a.1.1.2
      cf : Primrec fun a => a.2.1
      cg : Primrec fun a => a.2.2.1
      z : Primrec fun a => (Nat.unpair a.1.1.2).1
      L : Primrec fun a => a.1.1.1.1
      ⊢ Primrec fun p => (fun a y => Bind.bind (Nat.Partrec.Code.lup a.1.1.1 { fst : …
    -/
    have z := z.comp (Primrec.fst (β := ℕ))
    have y := Primrec.snd
      (α := ((List (List (Option ℕ)) × ℕ) × ℕ) × Code × Code × Option ℕ × Option ℕ) (β := ℕ)
    have h₁ := hlup.comp <| L.pair <| (((k'.pair c).comp Primrec.fst).comp Primrec.fst).pair
      (Primrec₂.natPair.comp z y)
    /-
      case hpc
      a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
      k✝¹ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.leng …
      n✝¹ : Primrec Prod.snd
      k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.len …
      n✝ : Primrec fun a => a.1.2
      k' : Primrec Prod.snd
      c : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      L✝ : Primrec fun a => a.1.1.1
      k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.1.le …
      n : Primrec fun a => a.1.1.2
      cf : Primrec fun a => a.2.1
      cg : Primrec fun a => a.2.2.1
      z✝ : Primrec fun a => (Nat.unpair a.1.1.2).1
      L : Primrec fun a => a.1.1.1.1
      z : Primrec fun a => (Nat.unpair a.1.1.1.2).1
      y : Primrec Prod.snd
      h₁ : Primrec fun a => Nat.Partrec.Code.lup { fst := a.1.1.1.1, snd := { fst := …
      ⊢ Primrec fun p => (fun a y => Bind.bind (Nat.Partrec.Code.lup a.1.1.1 { fst : …
    -/
    refine Primrec.option_bind h₁ (?_ : Primrec _)
    /-
      case hpc
      a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
      k✝¹ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.leng …
      n✝¹ : Primrec Prod.snd
      k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.len …
      n✝ : Primrec fun a => a.1.2
      k' : Primrec Prod.snd
      c : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      L✝ : Primrec fun a => a.1.1.1
      k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.1.le …
      n : Primrec fun a => a.1.1.2
      cf : Primrec fun a => a.2.1
      cg : Primrec fun a => a.2.2.1
      z✝ : Primrec fun a => (Nat.unpair a.1.1.2).1
      L : Primrec fun a => a.1.1.1.1
      z : Primrec fun a => (Nat.unpair a.1.1.1.2).1
      y : Primrec Prod.snd
      h₁ : Primrec fun a => Nat.Partrec.Code.lup { fst := a.1.1.1.1, snd := { fst := …
      ⊢ Primrec fun p => (fun p i => Nat.Partrec.Code.lup p.1.1.1.1 { fst := (Denume …
    -/
    have z := z.comp (Primrec.fst (β := ℕ))
    /-
      case hpc
      a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
      k✝¹ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.leng …
      n✝¹ : Primrec Prod.snd
      k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.len …
      n✝ : Primrec fun a => a.1.2
      k' : Primrec Prod.snd
      c : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      L✝ : Primrec fun a => a.1.1.1
      k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.1.le …
      n : Primrec fun a => a.1.1.2
      cf : Primrec fun a => a.2.1
      cg : Primrec fun a => a.2.2.1
      z✝¹ : Primrec fun a => (Nat.unpair a.1.1.2).1
      L : Primrec fun a => a.1.1.1.1
      z✝ : Primrec fun a => (Nat.unpair a.1.1.1.2).1
      y : Primrec Prod.snd
      h₁ : Primrec fun a => Nat.Partrec.Code.lup { fst := a.1.1.1.1, snd := { fst := …
      z : Primrec fun a => (Nat.unpair a.1.1.1.1.2).1
      ⊢ Primrec fun p => (fun p i => Nat.Partrec.Code.lup p.1.1.1.1 { fst := (Denume …
    -/
    have y := y.comp (Primrec.fst (β := ℕ))
    have i := Primrec.snd
      (α := (((List (List (Option ℕ)) × ℕ) × ℕ) × Code × Code × Option ℕ × Option ℕ) × ℕ)
      (β := ℕ)
    have h₂ := hlup.comp ((L.comp Primrec.fst).pair <|
      ((k.pair cg).comp <| Primrec.fst.comp Primrec.fst).pair <|
        Primrec₂.natPair.comp z <| Primrec₂.natPair.comp y i)
    /-
      case hpc
      a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
      k✝¹ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.leng …
      n✝¹ : Primrec Prod.snd
      k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.len …
      n✝ : Primrec fun a => a.1.2
      k' : Primrec Prod.snd
      c : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      L✝ : Primrec fun a => a.1.1.1
      k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.1.le …
      n : Primrec fun a => a.1.1.2
      cf : Primrec fun a => a.2.1
      cg : Primrec fun a => a.2.2.1
      z✝¹ : Primrec fun a => (Nat.unpair a.1.1.2).1
      L : Primrec fun a => a.1.1.1.1
      z✝ : Primrec fun a => (Nat.unpair a.1.1.1.2).1
      y✝ : Primrec Prod.snd
      h₁ : Primrec fun a => Nat.Partrec.Code.lup { fst := a.1.1.1.1, snd := { fst := …
      z : Primrec fun a => (Nat.unpair a.1.1.1.1.2).1
      y : Primrec fun a => a.1.2
      i : Primrec Prod.snd
      h₂ : Primrec fun a => Nat.Partrec.Code.lup { fst := a.1.1.1.1.1, snd := { fst  …
      ⊢ Primrec fun p => (fun p i => Nat.Partrec.Code.lup p.1.1.1.1 { fst := (Denume …
    -/
    exact h₂
    /-
      🎉 no goals
    -/
  · have L := (Primrec.fst.comp Primrec.fst).comp
      (Primrec.fst (α := (List (List (Option ℕ)) × ℕ) × ℕ)
        (β := Code × Option ℕ))
    /-
      case hrf
      a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
      k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.lengt …
      n✝ : Primrec Prod.snd
      k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      n : Primrec fun a => a.1.2
      k' : Primrec Prod.snd
      c : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      L : Primrec fun a => a.1.1.1
      ⊢ Primrec fun a =>
          let z := (Nat.unpair a.1.1.2).1;
          let m := (Nat.unpair a.1.1.2).2;
          Bind.bind (Nat.Partrec.Code.lup a.1.1.1 { fst := (Denumerable.ofNat (Prod  …
    -/
    have k := k.comp (Primrec.fst (β := Code × Option ℕ))
    /-
      case hrf
      a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
      k✝¹ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.leng …
      n✝ : Primrec Prod.snd
      k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.len …
      n : Primrec fun a => a.1.2
      k' : Primrec Prod.snd
      c : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      L : Primrec fun a => a.1.1.1
      k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.1.le …
      ⊢ Primrec fun a =>
          let z := (Nat.unpair a.1.1.2).1;
          let m := (Nat.unpair a.1.1.2).2;
          Bind.bind (Nat.Partrec.Code.lup a.1.1.1 { fst := (Denumerable.ofNat (Prod  …
    -/
    have n := n.comp (Primrec.fst (β := Code × Option ℕ))
    have cf := Primrec.fst.comp (Primrec.snd (α := (List (List (Option ℕ)) × ℕ) × ℕ)
        (β := Code × Option ℕ))
    /-
      case hrf
      a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
      k✝¹ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.leng …
      n✝¹ : Primrec Prod.snd
      k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.len …
      n✝ : Primrec fun a => a.1.2
      k' : Primrec Prod.snd
      c : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      L : Primrec fun a => a.1.1.1
      k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.1.le …
      n : Primrec fun a => a.1.1.2
      cf : Primrec fun a => a.2.1
      ⊢ Primrec fun a =>
          let z := (Nat.unpair a.1.1.2).1;
          let m := (Nat.unpair a.1.1.2).2;
          Bind.bind (Nat.Partrec.Code.lup a.1.1.1 { fst := (Denumerable.ofNat (Prod  …
    -/
    have z := Primrec.fst.comp (Primrec.unpair.comp n)
    /-
      case hrf
      a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
      k✝¹ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.leng …
      n✝¹ : Primrec Prod.snd
      k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.len …
      n✝ : Primrec fun a => a.1.2
      k' : Primrec Prod.snd
      c : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      L : Primrec fun a => a.1.1.1
      k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.1.le …
      n : Primrec fun a => a.1.1.2
      cf : Primrec fun a => a.2.1
      z : Primrec fun a => (Nat.unpair a.1.1.2).1
      ⊢ Primrec fun a =>
          let z := (Nat.unpair a.1.1.2).1;
          let m := (Nat.unpair a.1.1.2).2;
          Bind.bind (Nat.Partrec.Code.lup a.1.1.1 { fst := (Denumerable.ofNat (Prod  …
    -/
    have m := Primrec.snd.comp (Primrec.unpair.comp n)
    /-
      case hrf
      a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
      k✝¹ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.leng …
      n✝¹ : Primrec Prod.snd
      k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.len …
      n✝ : Primrec fun a => a.1.2
      k' : Primrec Prod.snd
      c : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      L : Primrec fun a => a.1.1.1
      k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.1.le …
      n : Primrec fun a => a.1.1.2
      cf : Primrec fun a => a.2.1
      z : Primrec fun a => (Nat.unpair a.1.1.2).1
      m : Primrec fun a => (Nat.unpair a.1.1.2).2
      ⊢ Primrec fun a =>
          let z := (Nat.unpair a.1.1.2).1;
          let m := (Nat.unpair a.1.1.2).2;
          Bind.bind (Nat.Partrec.Code.lup a.1.1.1 { fst := (Denumerable.ofNat (Prod  …
    -/
    have h₁ := hlup.comp <| L.pair <| (k.pair cf).pair (Primrec₂.natPair.comp z m)
    /-
      case hrf
      a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
      k✝¹ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.leng …
      n✝¹ : Primrec Prod.snd
      k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.len …
      n✝ : Primrec fun a => a.1.2
      k' : Primrec Prod.snd
      c : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      L : Primrec fun a => a.1.1.1
      k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.1.le …
      n : Primrec fun a => a.1.1.2
      cf : Primrec fun a => a.2.1
      z : Primrec fun a => (Nat.unpair a.1.1.2).1
      m : Primrec fun a => (Nat.unpair a.1.1.2).2
      h₁ : Primrec fun a => Nat.Partrec.Code.lup { fst := a.1.1.1, snd := { fst := { …
      ⊢ Primrec fun a =>
          let z := (Nat.unpair a.1.1.2).1;
          let m := (Nat.unpair a.1.1.2).2;
          Bind.bind (Nat.Partrec.Code.lup a.1.1.1 { fst := (Denumerable.ofNat (Prod  …
    -/
    refine Primrec.option_bind h₁ (?_ : Primrec _)
    /-
      case hrf
      a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
      k✝¹ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.leng …
      n✝¹ : Primrec Prod.snd
      k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.len …
      n✝ : Primrec fun a => a.1.2
      k' : Primrec Prod.snd
      c : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      L : Primrec fun a => a.1.1.1
      k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.1.le …
      n : Primrec fun a => a.1.1.2
      cf : Primrec fun a => a.2.1
      z : Primrec fun a => (Nat.unpair a.1.1.2).1
      m : Primrec fun a => (Nat.unpair a.1.1.2).2
      h₁ : Primrec fun a => Nat.Partrec.Code.lup { fst := a.1.1.1, snd := { fst := { …
      ⊢ Primrec fun p => (fun a x => Nat.casesOn x (Option.some (Nat.unpair a.1.1.2) …
    -/
    have m := m.comp (Primrec.fst (β := ℕ))
    /-
      case hrf
      a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
      k✝¹ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.leng …
      n✝¹ : Primrec Prod.snd
      k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.len …
      n✝ : Primrec fun a => a.1.2
      k' : Primrec Prod.snd
      c : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      L : Primrec fun a => a.1.1.1
      k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.1.le …
      n : Primrec fun a => a.1.1.2
      cf : Primrec fun a => a.2.1
      z : Primrec fun a => (Nat.unpair a.1.1.2).1
      m✝ : Primrec fun a => (Nat.unpair a.1.1.2).2
      h₁ : Primrec fun a => Nat.Partrec.Code.lup { fst := a.1.1.1, snd := { fst := { …
      m : Primrec fun a => (Nat.unpair a.1.1.1.2).2
      ⊢ Primrec fun p => (fun a x => Nat.casesOn x (Option.some (Nat.unpair a.1.1.2) …
    -/
    refine Primrec.nat_casesOn Primrec.snd (Primrec.option_some.comp m) ?_
    /-
      case hrf
      a : Primrec fun a => Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.length
      k✝¹ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.leng …
      n✝¹ : Primrec Prod.snd
      k✝ : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.len …
      n✝ : Primrec fun a => a.1.2
      k' : Primrec Prod.snd
      c : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.leng …
      L : Primrec fun a => a.1.1.1
      k : Primrec fun a => (Denumerable.ofNat (Prod Nat Nat.Partrec.Code) a.1.1.1.le …
      n : Primrec fun a => a.1.1.2
      cf : Primrec fun a => a.2.1
      z : Primrec fun a => (Nat.unpair a.1.1.2).1
      m✝ : Primrec fun a => (Nat.unpair a.1.1.2).2
      h₁ : Primrec fun a => Nat.Partrec.Code.lup { fst := a.1.1.1, snd := { fst := { …
      m : Primrec fun a => (Nat.unpair a.1.1.1.2).2
      ⊢ Primrec₂ fun p x => Nat.Partrec.Code.lup p.1.1.1.1 { fst := p.1.1.2, snd :=  …
    -/
    unfold Primrec₂
    exact (hlup.comp ((L.comp Primrec.fst).pair <|
      ((k'.pair c).comp <| Primrec.fst.comp Primrec.fst).pair
        (Primrec₂.natPair.comp (z.comp Primrec.fst) (_root_.Primrec.succ.comp m)))).comp
      Primrec.fst


private theorem evaln_map (k c n) :
    ((List.range k)[n]?.bind fun a ↦ evaln k c a) = evaln k c n := by
  /-
    k : Nat
    c : Nat.Partrec.Code
    n : Nat
    ⊢ Eq ((GetElem?.getElem? (List.range k) n).bind fun a => Nat.Partrec.Code.eval …
  -/
  by_cases kn : n < k
    /-
      case pos
      k : Nat
      c : Nat.Partrec.Code
      n : Nat
      kn : LT.lt n k
      ⊢ Eq ((GetElem?.getElem? (List.range k) n).bind fun a => Nat.Partrec.Code.eval …
    -/
  · simp [List.getElem?_range kn]
    /-
      🎉 no goals
    -/
    /-
      case neg
      k : Nat
      c : Nat.Partrec.Code
      n : Nat
      kn : Not (LT.lt n k)
      ⊢ Eq ((GetElem?.getElem? (List.range k) n).bind fun a => Nat.Partrec.Code.eval …
    -/
  · rw [List.getElem?_eq_none]
      /-
        case neg
        k : Nat
        c : Nat.Partrec.Code
        n : Nat
        kn : Not (LT.lt n k)
        ⊢ Eq (Option.none.bind fun a => Nat.Partrec.Code.evaln k c a) (Nat.Partrec.Cod …
      -/
    · cases e : evaln k c n
        /-
          case neg.none
          k : Nat
          c : Nat.Partrec.Code
          n : Nat
          kn : Not (LT.lt n k)
          e : Eq (Nat.Partrec.Code.evaln k c n) Option.none
          ⊢ Eq (Option.none.bind fun a => Nat.Partrec.Code.evaln k c a) Option.none
        -/
      · rfl
        /-
          🎉 no goals
        -/
      /-
        case neg.some
        k : Nat
        c : Nat.Partrec.Code
        n : Nat
        kn : Not (LT.lt n k)
        val✝ : Nat
        e : Eq (Nat.Partrec.Code.evaln k c n) (Option.some val✝)
        ⊢ Eq (Option.none.bind fun a => Nat.Partrec.Code.evaln k c a) (Option.some val✝)
      -/
      exact kn.elim (evaln_bound e)
      /-
        🎉 no goals
      -/
    /-
      case neg
      k : Nat
      c : Nat.Partrec.Code
      n : Nat
      kn : Not (LT.lt n k)
      ⊢ LE.le (List.range k).length n
    -/
    simpa using kn
    /-
      🎉 no goals
    -/


/-- The `Nat.Partrec.Code.evaln` function is primitive recursive. -/
theorem evaln_prim : Primrec fun a : (ℕ × Code) × ℕ => evaln a.1.1 a.1.2 a.2 :=
  have :
    Primrec₂ fun (_ : Unit) (n : ℕ) =>
      let a := ofNat (ℕ × Code) n
      (List.range a.1).map (evaln a.1 a.2) :=
    Primrec.nat_strong_rec _ (hG.comp Primrec.snd).to₂ fun _ p => by
      simp only [G, prod_ofNat_val, ofNat_nat, List.length_map, List.length_range,
        Nat.pair_unpair, Option.some_inj]
      /-
        x✝ : Unit
        p : Nat
        ⊢ Eq (List.map (fun n => Nat.rec Option.none (fun n_1 n_ih => Nat.Partrec.Code …
      -/
      refine List.map_congr_left fun n => ?_
      have : List.range p = List.range (Nat.pair p.unpair.1 (encode (ofNat Code p.unpair.2))) := by
        simp
      /-
        x✝ : Unit
        p n : Nat
        this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
        ⊢ Membership.mem (List.range (Nat.unpair p).1) n → Eq (Nat.rec Option.none (fu …
      -/
      rw [this]
      /-
        x✝ : Unit
        p n : Nat
        this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
        ⊢ Membership.mem (List.range (Nat.unpair p).1) n → Eq (Nat.rec Option.none (fu …
      -/
      generalize p.unpair.1 = k
      /-
        x✝ : Unit
        p n : Nat
        this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
        k : Nat
        ⊢ Membership.mem (List.range k) n → Eq (Nat.rec Option.none (fun n_1 n_ih => N …
      -/
      generalize ofNat Code p.unpair.2 = c
      /-
        x✝ : Unit
        p n : Nat
        this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
        k : Nat
        c : Nat.Partrec.Code
        ⊢ Membership.mem (List.range k) n → Eq (Nat.rec Option.none (fun n_1 n_ih => N …
      -/
      intro nk
      /-
        x✝ : Unit
        p n : Nat
        this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
        k : Nat
        c : Nat.Partrec.Code
        nk : Membership.mem (List.range k) n
        ⊢ Eq (Nat.rec Option.none (fun n_1 n_ih => Nat.Partrec.Code.rec (Option.some 0 …
      -/
      cases' k with k'
        /-
          case zero
          x✝ : Unit
          p n : Nat
          this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
          c : Nat.Partrec.Code
          nk : Membership.mem (List.range 0) n
          ⊢ Eq (Nat.rec Option.none (fun n_1 n_ih => Nat.Partrec.Code.rec (Option.some 0 …
        -/
      · simp [evaln]
        /-
          🎉 no goals
        -/
      /-
        case succ
        x✝ : Unit
        p n : Nat
        this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
        c : Nat.Partrec.Code
        k' : Nat
        nk : Membership.mem (List.range (HAdd.hAdd k' 1)) n
        ⊢ Eq (Nat.rec Option.none (fun n_1 n_ih => Nat.Partrec.Code.rec (Option.some 0 …
      -/
      let k := k' + 1
      /-
        case succ
        x✝ : Unit
        p n : Nat
        this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
        c : Nat.Partrec.Code
        k' : Nat
        nk : Membership.mem (List.range (HAdd.hAdd k' 1)) n
        k : Nat := HAdd.hAdd k' 1
        ⊢ Eq (Nat.rec Option.none (fun n_1 n_ih => Nat.Partrec.Code.rec (Option.some 0 …
      -/
      simp only [show k'.succ = k from rfl]
      /-
        case succ
        x✝ : Unit
        p n : Nat
        this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
        c : Nat.Partrec.Code
        k' : Nat
        nk : Membership.mem (List.range (HAdd.hAdd k' 1)) n
        k : Nat := HAdd.hAdd k' 1
        ⊢ Eq (Nat.Partrec.Code.rec (Option.some 0) (Option.some n.succ) (Option.some ( …
      -/
      simp? [Nat.lt_succ_iff] at nk says simp only [List.mem_range, Nat.lt_succ_iff] at nk
      have hg :
        ∀ {k' c' n},
          Nat.pair k' (encode c') < Nat.pair k (encode c) →
            lup ((List.range (Nat.pair k (encode c))).map fun n =>
              (List.range n.unpair.1).map (evaln n.unpair.1 (ofNat Code n.unpair.2))) (k', c') n =
            evaln k' c' n := by
        intro k₁ c₁ n₁ hl
        simp [lup, List.getElem?_range hl, evaln_map, Bind.bind, Option.bind_map]
      /-
        case succ
        x✝ : Unit
        p n : Nat
        this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
        c : Nat.Partrec.Code
        k' : Nat
        k : Nat := HAdd.hAdd k' 1
        nk : LE.le n k'
        hg : ∀ {k' : Nat} {c' : Nat.Partrec.Code} {n : Nat}, LT.lt (Nat.pair k' (Encod …
        ⊢ Eq (Nat.Partrec.Code.rec (Option.some 0) (Option.some n.succ) (Option.some ( …
      -/
      cases' c with cf cg cf cg cf cg cf <;>
        /-
          case succ.zero
          x✝ : Unit
          p n : Nat
          this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
          k' : Nat
          k : Nat := HAdd.hAdd k' 1
          nk : LE.le n k'
          hg : ∀ {k' : Nat} {c' : Nat.Partrec.Code} {n : Nat}, LT.lt (Nat.pair k' (Encod …
          ⊢ Eq (Nat.Partrec.Code.rec (Option.some 0) (Option.some n.succ) (Option.some ( …
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
          🎉 no goals
        -/
        simp [evaln, nk, Bind.bind, Functor.map, Seq.seq, pure]
        /-
          case succ.pair
          x✝ : Unit
          p n : Nat
          this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
          k' : Nat
          k : Nat := HAdd.hAdd k' 1
          nk : LE.le n k'
          cf cg : Nat.Partrec.Code
          hg : ∀ {k' : Nat} {c' : Nat.Partrec.Code} {n : Nat}, LT.lt (Nat.pair k' (Encod …
          ⊢ Eq ((Nat.Partrec.Code.lup (List.map (fun n => List.map (Nat.Partrec.Code.eva …
        -/
      · cases' encode_lt_pair cf cg with lf lg
        /-
          case succ.pair.intro
          x✝ : Unit
          p n : Nat
          this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
          k' : Nat
          k : Nat := HAdd.hAdd k' 1
          nk : LE.le n k'
          cf cg : Nat.Partrec.Code
          hg : ∀ {k' : Nat} {c' : Nat.Partrec.Code} {n : Nat}, LT.lt (Nat.pair k' (Encod …
          lf : LT.lt (Encodable.encode cf) (Encodable.encode (cf.pair cg))
          lg : LT.lt (Encodable.encode cg) (Encodable.encode (cf.pair cg))
          ⊢ Eq ((Nat.Partrec.Code.lup (List.map (fun n => List.map (Nat.Partrec.Code.eva …
        -/
        rw [hg (Nat.pair_lt_pair_right _ lf), hg (Nat.pair_lt_pair_right _ lg)]
        /-
          case succ.pair.intro
          x✝ : Unit
          p n : Nat
          this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
          k' : Nat
          k : Nat := HAdd.hAdd k' 1
          nk : LE.le n k'
          cf cg : Nat.Partrec.Code
          hg : ∀ {k' : Nat} {c' : Nat.Partrec.Code} {n : Nat}, LT.lt (Nat.pair k' (Encod …
          lf : LT.lt (Encodable.encode cf) (Encodable.encode (cf.pair cg))
          lg : LT.lt (Encodable.encode cg) (Encodable.encode (cf.pair cg))
          ⊢ Eq ((Nat.Partrec.Code.evaln k cf n).bind fun a => (Nat.Partrec.Code.evaln k  …
        -/
        cases evaln k cf n
          /-
            case succ.pair.intro.none
            x✝ : Unit
            p n : Nat
            this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
            k' : Nat
            k : Nat := HAdd.hAdd k' 1
            nk : LE.le n k'
            cf cg : Nat.Partrec.Code
            hg : ∀ {k' : Nat} {c' : Nat.Partrec.Code} {n : Nat}, LT.lt (Nat.pair k' (Encod …
            lf : LT.lt (Encodable.encode cf) (Encodable.encode (cf.pair cg))
            lg : LT.lt (Encodable.encode cg) (Encodable.encode (cf.pair cg))
            ⊢ Eq (Option.none.bind fun a => (Nat.Partrec.Code.evaln k cg n).bind fun y =>  …
          -/
        · rfl
          /-
            🎉 no goals
          -/
        /-
          case succ.pair.intro.some
          x✝ : Unit
          p n : Nat
          this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
          k' : Nat
          k : Nat := HAdd.hAdd k' 1
          nk : LE.le n k'
          cf cg : Nat.Partrec.Code
          hg : ∀ {k' : Nat} {c' : Nat.Partrec.Code} {n : Nat}, LT.lt (Nat.pair k' (Encod …
          lf : LT.lt (Encodable.encode cf) (Encodable.encode (cf.pair cg))
          lg : LT.lt (Encodable.encode cg) (Encodable.encode (cf.pair cg))
          val✝ : Nat
          ⊢ Eq ((Option.some val✝).bind fun a => (Nat.Partrec.Code.evaln k cg n).bind fu …
        -/
                               /-
                                 🎉 no goals
                               -/
        cases evaln k cg n <;> rfl
                               /-
                                 🎉 no goals
                               -/
        /-
          case succ.comp
          x✝ : Unit
          p n : Nat
          this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
          k' : Nat
          k : Nat := HAdd.hAdd k' 1
          nk : LE.le n k'
          cf cg : Nat.Partrec.Code
          hg : ∀ {k' : Nat} {c' : Nat.Partrec.Code} {n : Nat}, LT.lt (Nat.pair k' (Encod …
          ⊢ Eq ((Nat.Partrec.Code.lup (List.map (fun n => List.map (Nat.Partrec.Code.eva …
        -/
      · cases' encode_lt_comp cf cg with lf lg
        /-
          case succ.comp.intro
          x✝ : Unit
          p n : Nat
          this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
          k' : Nat
          k : Nat := HAdd.hAdd k' 1
          nk : LE.le n k'
          cf cg : Nat.Partrec.Code
          hg : ∀ {k' : Nat} {c' : Nat.Partrec.Code} {n : Nat}, LT.lt (Nat.pair k' (Encod …
          lf : LT.lt (Encodable.encode cf) (Encodable.encode (cf.comp cg))
          lg : LT.lt (Encodable.encode cg) (Encodable.encode (cf.comp cg))
          ⊢ Eq ((Nat.Partrec.Code.lup (List.map (fun n => List.map (Nat.Partrec.Code.eva …
        -/
        rw [hg (Nat.pair_lt_pair_right _ lg)]
        /-
          case succ.comp.intro
          x✝ : Unit
          p n : Nat
          this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
          k' : Nat
          k : Nat := HAdd.hAdd k' 1
          nk : LE.le n k'
          cf cg : Nat.Partrec.Code
          hg : ∀ {k' : Nat} {c' : Nat.Partrec.Code} {n : Nat}, LT.lt (Nat.pair k' (Encod …
          lf : LT.lt (Encodable.encode cf) (Encodable.encode (cf.comp cg))
          lg : LT.lt (Encodable.encode cg) (Encodable.encode (cf.comp cg))
          ⊢ Eq ((Nat.Partrec.Code.evaln k cg n).bind fun x => Nat.Partrec.Code.lup (List …
        -/
        cases evaln k cg n
          /-
            case succ.comp.intro.none
            x✝ : Unit
            p n : Nat
            this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
            k' : Nat
            k : Nat := HAdd.hAdd k' 1
            nk : LE.le n k'
            cf cg : Nat.Partrec.Code
            hg : ∀ {k' : Nat} {c' : Nat.Partrec.Code} {n : Nat}, LT.lt (Nat.pair k' (Encod …
            lf : LT.lt (Encodable.encode cf) (Encodable.encode (cf.comp cg))
            lg : LT.lt (Encodable.encode cg) (Encodable.encode (cf.comp cg))
            ⊢ Eq (Option.none.bind fun x => Nat.Partrec.Code.lup (List.map (fun n => List. …
          -/
        · rfl
          /-
            🎉 no goals
          -/
        /-
          case succ.comp.intro.some
          x✝ : Unit
          p n : Nat
          this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
          k' : Nat
          k : Nat := HAdd.hAdd k' 1
          nk : LE.le n k'
          cf cg : Nat.Partrec.Code
          hg : ∀ {k' : Nat} {c' : Nat.Partrec.Code} {n : Nat}, LT.lt (Nat.pair k' (Encod …
          lf : LT.lt (Encodable.encode cf) (Encodable.encode (cf.comp cg))
          lg : LT.lt (Encodable.encode cg) (Encodable.encode (cf.comp cg))
          val✝ : Nat
          ⊢ Eq ((Option.some val✝).bind fun x => Nat.Partrec.Code.lup (List.map (fun n = …
        -/
        simp [k, hg (Nat.pair_lt_pair_right _ lf)]
        /-
          🎉 no goals
        -/
        /-
          case succ.prec
          x✝ : Unit
          p n : Nat
          this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
          k' : Nat
          k : Nat := HAdd.hAdd k' 1
          nk : LE.le n k'
          cf cg : Nat.Partrec.Code
          hg : ∀ {k' : Nat} {c' : Nat.Partrec.Code} {n : Nat}, LT.lt (Nat.pair k' (Encod …
          ⊢ Eq (Nat.rec (Nat.Partrec.Code.lup (List.map (fun n => List.map (Nat.Partrec. …
        -/
      · cases' encode_lt_prec cf cg with lf lg
        /-
          case succ.prec.intro
          x✝ : Unit
          p n : Nat
          this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
          k' : Nat
          k : Nat := HAdd.hAdd k' 1
          nk : LE.le n k'
          cf cg : Nat.Partrec.Code
          hg : ∀ {k' : Nat} {c' : Nat.Partrec.Code} {n : Nat}, LT.lt (Nat.pair k' (Encod …
          lf : LT.lt (Encodable.encode cf) (Encodable.encode (cf.prec cg))
          lg : LT.lt (Encodable.encode cg) (Encodable.encode (cf.prec cg))
          ⊢ Eq (Nat.rec (Nat.Partrec.Code.lup (List.map (fun n => List.map (Nat.Partrec. …
        -/
        rw [hg (Nat.pair_lt_pair_right _ lf)]
        /-
          case succ.prec.intro
          x✝ : Unit
          p n : Nat
          this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
          k' : Nat
          k : Nat := HAdd.hAdd k' 1
          nk : LE.le n k'
          cf cg : Nat.Partrec.Code
          hg : ∀ {k' : Nat} {c' : Nat.Partrec.Code} {n : Nat}, LT.lt (Nat.pair k' (Encod …
          lf : LT.lt (Encodable.encode cf) (Encodable.encode (cf.prec cg))
          lg : LT.lt (Encodable.encode cg) (Encodable.encode (cf.prec cg))
          ⊢ Eq (Nat.rec (Nat.Partrec.Code.evaln k cf (Nat.unpair n).1) (fun n_1 n_ih =>  …
        -/
        cases n.unpair.2
          /-
            case succ.prec.intro.zero
            x✝ : Unit
            p n : Nat
            this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
            k' : Nat
            k : Nat := HAdd.hAdd k' 1
            nk : LE.le n k'
            cf cg : Nat.Partrec.Code
            hg : ∀ {k' : Nat} {c' : Nat.Partrec.Code} {n : Nat}, LT.lt (Nat.pair k' (Encod …
            lf : LT.lt (Encodable.encode cf) (Encodable.encode (cf.prec cg))
            lg : LT.lt (Encodable.encode cg) (Encodable.encode (cf.prec cg))
            ⊢ Eq (Nat.rec (Nat.Partrec.Code.evaln k cf (Nat.unpair n).1) (fun n_1 n_ih =>  …
          -/
        · rfl
          /-
            🎉 no goals
          -/
        /-
          case succ.prec.intro.succ
          x✝ : Unit
          p n : Nat
          this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
          k' : Nat
          k : Nat := HAdd.hAdd k' 1
          nk : LE.le n k'
          cf cg : Nat.Partrec.Code
          hg : ∀ {k' : Nat} {c' : Nat.Partrec.Code} {n : Nat}, LT.lt (Nat.pair k' (Encod …
          lf : LT.lt (Encodable.encode cf) (Encodable.encode (cf.prec cg))
          lg : LT.lt (Encodable.encode cg) (Encodable.encode (cf.prec cg))
          n✝ : Nat
          ⊢ Eq (Nat.rec (Nat.Partrec.Code.evaln k cf (Nat.unpair n).1) (fun n_1 n_ih =>  …
        -/
        simp only [decode_eq_ofNat, Option.some.injEq]
        /-
          case succ.prec.intro.succ
          x✝ : Unit
          p n : Nat
          this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
          k' : Nat
          k : Nat := HAdd.hAdd k' 1
          nk : LE.le n k'
          cf cg : Nat.Partrec.Code
          hg : ∀ {k' : Nat} {c' : Nat.Partrec.Code} {n : Nat}, LT.lt (Nat.pair k' (Encod …
          lf : LT.lt (Encodable.encode cf) (Encodable.encode (cf.prec cg))
          lg : LT.lt (Encodable.encode cg) (Encodable.encode (cf.prec cg))
          n✝ : Nat
          ⊢ Eq ((Nat.Partrec.Code.lup (List.map (fun n => List.map (Nat.Partrec.Code.eva …
        -/
        rw [hg (Nat.pair_lt_pair_left _ k'.lt_succ_self)]
        /-
          case succ.prec.intro.succ
          x✝ : Unit
          p n : Nat
          this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
          k' : Nat
          k : Nat := HAdd.hAdd k' 1
          nk : LE.le n k'
          cf cg : Nat.Partrec.Code
          hg : ∀ {k' : Nat} {c' : Nat.Partrec.Code} {n : Nat}, LT.lt (Nat.pair k' (Encod …
          lf : LT.lt (Encodable.encode cf) (Encodable.encode (cf.prec cg))
          lg : LT.lt (Encodable.encode cg) (Encodable.encode (cf.prec cg))
          n✝ : Nat
          ⊢ Eq ((Nat.Partrec.Code.evaln k' (cf.prec cg) (Nat.pair (Nat.unpair n).1 n✝)). …
        -/
        cases evaln k' _ _
          /-
            case succ.prec.intro.succ.none
            x✝ : Unit
            p n : Nat
            this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
            k' : Nat
            k : Nat := HAdd.hAdd k' 1
            nk : LE.le n k'
            cf cg : Nat.Partrec.Code
            hg : ∀ {k' : Nat} {c' : Nat.Partrec.Code} {n : Nat}, LT.lt (Nat.pair k' (Encod …
            lf : LT.lt (Encodable.encode cf) (Encodable.encode (cf.prec cg))
            lg : LT.lt (Encodable.encode cg) (Encodable.encode (cf.prec cg))
            n✝ : Nat
            ⊢ Eq (Option.none.bind fun i => Nat.Partrec.Code.lup (List.map (fun n => List. …
          -/
        · rfl
          /-
            🎉 no goals
          -/
        /-
          case succ.prec.intro.succ.some
          x✝ : Unit
          p n : Nat
          this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
          k' : Nat
          k : Nat := HAdd.hAdd k' 1
          nk : LE.le n k'
          cf cg : Nat.Partrec.Code
          hg : ∀ {k' : Nat} {c' : Nat.Partrec.Code} {n : Nat}, LT.lt (Nat.pair k' (Encod …
          lf : LT.lt (Encodable.encode cf) (Encodable.encode (cf.prec cg))
          lg : LT.lt (Encodable.encode cg) (Encodable.encode (cf.prec cg))
          n✝ val✝ : Nat
          ⊢ Eq ((Option.some val✝).bind fun i => Nat.Partrec.Code.lup (List.map (fun n = …
        -/
        simp [k, hg (Nat.pair_lt_pair_right _ lg)]
        /-
          🎉 no goals
        -/
        /-
          case succ.rfind'
          x✝ : Unit
          p n : Nat
          this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
          k' : Nat
          k : Nat := HAdd.hAdd k' 1
          nk : LE.le n k'
          cf : Nat.Partrec.Code
          hg : ∀ {k' : Nat} {c' : Nat.Partrec.Code} {n : Nat}, LT.lt (Nat.pair k' (Encod …
          ⊢ Eq ((Nat.Partrec.Code.lup (List.map (fun n => List.map (Nat.Partrec.Code.eva …
        -/
      · have lf := encode_lt_rfind' cf
        /-
          case succ.rfind'
          x✝ : Unit
          p n : Nat
          this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
          k' : Nat
          k : Nat := HAdd.hAdd k' 1
          nk : LE.le n k'
          cf : Nat.Partrec.Code
          hg : ∀ {k' : Nat} {c' : Nat.Partrec.Code} {n : Nat}, LT.lt (Nat.pair k' (Encod …
          lf : LT.lt (Encodable.encode cf) (Encodable.encode cf.rfind')
          ⊢ Eq ((Nat.Partrec.Code.lup (List.map (fun n => List.map (Nat.Partrec.Code.eva …
        -/
        rw [hg (Nat.pair_lt_pair_right _ lf)]
        /-
          case succ.rfind'
          x✝ : Unit
          p n : Nat
          this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
          k' : Nat
          k : Nat := HAdd.hAdd k' 1
          nk : LE.le n k'
          cf : Nat.Partrec.Code
          hg : ∀ {k' : Nat} {c' : Nat.Partrec.Code} {n : Nat}, LT.lt (Nat.pair k' (Encod …
          lf : LT.lt (Encodable.encode cf) (Encodable.encode cf.rfind')
          ⊢ Eq ((Nat.Partrec.Code.evaln k cf n).bind fun x => Nat.rec (Option.some (Nat. …
        -/
        cases' evaln k cf n with x
          /-
            case succ.rfind'.none
            x✝ : Unit
            p n : Nat
            this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
            k' : Nat
            k : Nat := HAdd.hAdd k' 1
            nk : LE.le n k'
            cf : Nat.Partrec.Code
            hg : ∀ {k' : Nat} {c' : Nat.Partrec.Code} {n : Nat}, LT.lt (Nat.pair k' (Encod …
            lf : LT.lt (Encodable.encode cf) (Encodable.encode cf.rfind')
            ⊢ Eq (Option.none.bind fun x => Nat.rec (Option.some (Nat.unpair n).2) (fun n_ …
          -/
        · rfl
          /-
            🎉 no goals
          -/
        /-
          case succ.rfind'.some
          x✝ : Unit
          p n : Nat
          this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
          k' : Nat
          k : Nat := HAdd.hAdd k' 1
          nk : LE.le n k'
          cf : Nat.Partrec.Code
          hg : ∀ {k' : Nat} {c' : Nat.Partrec.Code} {n : Nat}, LT.lt (Nat.pair k' (Encod …
          lf : LT.lt (Encodable.encode cf) (Encodable.encode cf.rfind')
          x : Nat
          ⊢ Eq ((Option.some x).bind fun x => Nat.rec (Option.some (Nat.unpair n).2) (fu …
        -/
        simp only [decode_eq_ofNat, Option.some.injEq, Option.some_bind]
        /-
          case succ.rfind'.some
          x✝ : Unit
          p n : Nat
          this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
          k' : Nat
          k : Nat := HAdd.hAdd k' 1
          nk : LE.le n k'
          cf : Nat.Partrec.Code
          hg : ∀ {k' : Nat} {c' : Nat.Partrec.Code} {n : Nat}, LT.lt (Nat.pair k' (Encod …
          lf : LT.lt (Encodable.encode cf) (Encodable.encode cf.rfind')
          x : Nat
          ⊢ Eq (Nat.rec (Option.some (Nat.unpair n).2) (fun n_1 n_ih => Nat.Partrec.Code …
        -/
                    /-
                      🎉 no goals
                    -/
        cases x <;> simp [Nat.succ_ne_zero]
        /-
          case succ.rfind'.some.succ
          x✝ : Unit
          p n : Nat
          this : Eq (List.range p) (List.range (Nat.pair (Nat.unpair p).1 (Encodable.enc …
          k' : Nat
          k : Nat := HAdd.hAdd k' 1
          nk : LE.le n k'
          cf : Nat.Partrec.Code
          hg : ∀ {k' : Nat} {c' : Nat.Partrec.Code} {n : Nat}, LT.lt (Nat.pair k' (Encod …
          lf : LT.lt (Encodable.encode cf) (Encodable.encode cf.rfind')
          n✝ : Nat
          ⊢ Eq (Nat.Partrec.Code.lup (List.map (fun n => List.map (Nat.Partrec.Code.eval …
        -/
        rw [hg (Nat.pair_lt_pair_left _ k'.lt_succ_self)]
        /-
          🎉 no goals
        -/
  (Primrec.option_bind
    (Primrec.list_get?.comp (this.comp (_root_.Primrec.const ())
      (Primrec.encode_iff.2 Primrec.fst)) Primrec.snd) Primrec.snd.to₂).of_eq
                          /-
                            this :
                              Primrec₂ fun x n =>
                                let a := Denumerable.ofNat (Prod Nat Nat.Partrec.Code) n;
                                List.map (Nat.Partrec.Code.evaln a.1 a.2) (List.range a.1)
                            x✝ : Prod (Prod Nat Nat.Partrec.Code) Nat
                            k : Nat
                            c : Nat.Partrec.Code
                            n : Nat
                            ⊢ Eq
                                (((let a := Denumerable.ofNat (Prod Nat Nat.Partrec.Code) (Encodable.encod …
                                        List.map (Nat.Partrec.Code.evaln a.1 a.2) (List.range a.1)).get?
                                      { fst := { fst := k, snd := c }, snd := n }.2).bind
                                  fun b => { fst := { fst := { fst := k, snd := c }, snd := n }, snd := b  …
                                (Nat.Partrec.Code.evaln { fst := { fst := k, snd := c }, snd := n }.1.1 {  …
                          -/
    fun ⟨⟨k, c⟩, n⟩ => by simp [evaln_map, Option.bind_map]
                          /-
                            🎉 no goals
                          -/


theorem eval_eq_rfindOpt (c n) : eval c n = Nat.rfindOpt fun k => evaln k c n :=
  Part.ext fun x => by
    /-
      c : Nat.Partrec.Code
      n x : Nat
      ⊢ Iff (Membership.mem (c.eval n) x) (Membership.mem (Nat.rfindOpt fun k => Nat …
    -/
    refine evaln_complete.trans (Nat.rfindOpt_mono ?_).symm
    /-
      c : Nat.Partrec.Code
      n x : Nat
      ⊢ ∀ {a m n_1 : Nat}, LE.le m n_1 → Membership.mem (Nat.Partrec.Code.evaln m c  …
    -/
    intro a m n hl; apply evaln_mono hl
                    /-
                      🎉 no goals
                    -/


theorem eval_part : Partrec₂ eval :=
  (Partrec.rfindOpt
    (evaln_prim.to_comp.comp ((Computable.snd.pair (fst.comp fst)).pair (snd.comp fst))).to₂).of_eq
                /-
                  a : Prod Nat.Partrec.Code Nat
                  ⊢ Eq (Nat.rfindOpt fun b => Nat.Partrec.Code.evaln { fst := { fst := { fst :=  …
                -/
    fun a => by simp [eval_eq_rfindOpt]
                /-
                  🎉 no goals
                -/


/-- Roger's fixed-point theorem: Any total, computable `f` has a fixed point: That is, under the
interpretation given by `Nat.Partrec.Code.eval`, there is a code `c` such that `c` and `f c` have
the same evaluation.
-/
theorem fixed_point {f : Code → Code} (hf : Computable f) : ∃ c : Code, eval (f c) = eval c :=
  let g (x y : ℕ) : Part ℕ := eval (ofNat Code x) x >>= fun b => eval (ofNat Code b) y
  have : Partrec₂ g :=
    (eval_part.comp ((Computable.ofNat _).comp fst) fst).bind
      (eval_part.comp ((Computable.ofNat _).comp snd) (snd.comp fst)).to₂
  let ⟨cg, eg⟩ := exists_code.1 this
                                                                           /-
                                                                             f : Nat.Partrec.Code → Nat.Partrec.Code
                                                                             hf : Computable f
                                                                             g : Nat → Nat → Part Nat := fun x y => Bind.bind ((Denumerable.ofNat Nat.Partr …
                                                                             this : Partrec₂ g
                                                                             cg : Nat.Partrec.Code
                                                                             eg : Eq cg.eval fun n => (↑(Encodable.decode n)).bind fun a => Part.map Encoda …
                                                                             ⊢ ∀ (a n : Nat), Eq (cg.eval (Nat.pair a n)) (Part.map Encodable.encode (g a n))
                                                                           -/
  have eg' : ∀ a n, eval cg (Nat.pair a n) = Part.map encode (g a n) := by simp [eg]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
  let F (x : ℕ) : Code := f (curry cg x)
  have : Computable F :=
    hf.comp (curry_prim.comp (_root_.Primrec.const cg) _root_.Primrec.id).to_comp
  let ⟨cF, eF⟩ := exists_code.1 this
                                                                            /-
                                                                              f : Nat.Partrec.Code → Nat.Partrec.Code
                                                                              hf : Computable f
                                                                              g : Nat → Nat → Part Nat := fun x y => Bind.bind ((Denumerable.ofNat Nat.Partr …
                                                                              this✝ : Partrec₂ g
                                                                              cg : Nat.Partrec.Code
                                                                              eg : Eq cg.eval fun n => (↑(Encodable.decode n)).bind fun a => Part.map Encoda …
                                                                              eg' : ∀ (a n : Nat), Eq (cg.eval (Nat.pair a n)) (Part.map Encodable.encode (g …
                                                                              F : Nat → Nat.Partrec.Code := fun x => f (cg.curry x)
                                                                              this : Computable F
                                                                              cF : Nat.Partrec.Code
                                                                              eF : Eq cF.eval fun n => (↑(Encodable.decode n)).bind fun a => Part.map Encoda …
                                                                              ⊢ Eq (cF.eval (Encodable.encode cF)) (Part.some (Encodable.encode (F (Encodabl …
                                                                            -/
  have eF' : eval cF (encode cF) = Part.some (encode (F (encode cF))) := by simp [eF]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
  ⟨curry cg (encode cF),
    funext fun n =>
      show eval (f (curry cg (encode cF))) n = eval (curry cg (encode cF)) n by
        /-
          f : Nat.Partrec.Code → Nat.Partrec.Code
          hf : Computable f
          g : Nat → Nat → Part Nat := fun x y => Bind.bind ((Denumerable.ofNat Nat.Partr …
          this✝ : Partrec₂ g
          cg : Nat.Partrec.Code
          eg : Eq cg.eval fun n => (↑(Encodable.decode n)).bind fun a => Part.map Encoda …
          eg' : ∀ (a n : Nat), Eq (cg.eval (Nat.pair a n)) (Part.map Encodable.encode (g …
          F : Nat → Nat.Partrec.Code := fun x => f (cg.curry x)
          this : Computable F
          cF : Nat.Partrec.Code
          eF : Eq cF.eval fun n => (↑(Encodable.decode n)).bind fun a => Part.map Encoda …
          eF' : Eq (cF.eval (Encodable.encode cF)) (Part.some (Encodable.encode (F (Enco …
          n : Nat
          ⊢ Eq ((f (cg.curry (Encodable.encode cF))).eval n) ((cg.curry (Encodable.encod …
        -/
        simp [F, g, eg', eF', Part.map_id']⟩
        /-
          🎉 no goals
        -/


theorem fixed_point₂ {f : Code → ℕ →. ℕ} (hf : Partrec₂ f) : ∃ c : Code, eval c = f c :=
  let ⟨cf, ef⟩ := exists_code.1 hf
  (fixed_point (curry_prim.comp (_root_.Primrec.const cf) Primrec.encode).to_comp).imp fun c e =>
                       /-
                         f : Nat.Partrec.Code → PFun Nat Nat
                         hf : Partrec₂ f
                         cf : Nat.Partrec.Code
                         ef : Eq cf.eval fun n => (↑(Encodable.decode n)).bind fun a => Part.map Encoda …
                         c : Nat.Partrec.Code
                         e : Eq (cf.curry (Encodable.encode c)).eval c.eval
                         n : Nat
                         ⊢ Eq (c.eval n) (f c n)
                       -/
    funext fun n => by simp [e.symm, ef, Part.map_id']
                       /-
                         🎉 no goals
                       -/


/-- There are only countably many partial recursive partial functions `ℕ →. ℕ`. -/
instance : Countable {f : ℕ →. ℕ // _root_.Partrec f} := by
  /-
    ⊢ Countable (Subtype fun f => _root_.Partrec f)
  -/
  apply Function.Surjective.countable (f := fun c => ⟨eval c, eval_part.comp (.const c) .id⟩)
  /-
    ⊢ Function.Surjective fun c => ⟨c.eval, ⋯⟩
  -/
  intro ⟨f, hf⟩; simpa using exists_code.1 hf
                 /-
                   🎉 no goals
                 -/


/-- There are only countably many computable functions `ℕ → ℕ`. -/
instance : Countable {f : ℕ → ℕ // Computable f} :=
  @Function.Injective.countable {f : ℕ → ℕ // Computable f} {f : ℕ →. ℕ // _root_.Partrec f} _
    (fun f => ⟨f.val, f.2⟩)
                                                             /-
                                                               x✝¹ x✝ : Subtype fun f => Computable f
                                                               h : Eq ((fun f => ⟨↑↑f, ⋯⟩) x✝¹) ((fun f => ⟨↑↑f, ⋯⟩) x✝)
                                                               ⊢ Eq ↑↑x✝¹ ↑↑x✝
                                                             -/
    (fun _ _ h => Subtype.val_inj.1 (PFun.lift_injective (by simpa using h)))
                                                             /-
                                                               🎉 no goals
                                                             -/


