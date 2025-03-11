/-- An encoding of a type in a certain alphabet, together with a decoding. -/
structure Encoding (α : Type u) where
  Γ : Type v
  encode : α → List Γ
  decode : List Γ → Option α
  decode_encode : ∀ x, decode (encode x) = some x


theorem Encoding.encode_injective {α : Type u} (e : Encoding α) : Function.Injective e.encode := by
  /-
    α : Type u
    e : Computability.Encoding α
    ⊢ Function.Injective e.encode
  -/
  refine fun _ _ h => Option.some_injective _ ?_
  /-
    α : Type u
    e : Computability.Encoding α
    x✝¹ x✝ : α
    h : Eq (e.encode x✝¹) (e.encode x✝)
    ⊢ Eq (Option.some x✝¹) (Option.some x✝)
  -/
  rw [← e.decode_encode, ← e.decode_encode, h]
  /-
    🎉 no goals
  -/


/-- An encoding plus a guarantee of finiteness of the alphabet. -/
structure FinEncoding (α : Type u) extends Encoding.{u, 0} α where
  ΓFin : Fintype Γ


instance Γ.fintype {α : Type u} (e : FinEncoding α) : Fintype e.toEncoding.Γ :=
  e.ΓFin


/-- A standard Turing machine alphabet, consisting of blank,bit0,bit1,bra,ket,comma. -/
inductive Γ'
  | blank
  | bit (b : Bool)
  | bra
  | ket
  | comma
  deriving DecidableEq

-- Porting note: A handler for `Fintype` had not been implemented yet.

instance Γ'.fintype : Fintype Γ' :=
                                                            /-
                                                              ⊢ (Insert.insert Computability.Γ'.blank (Insert.insert (Computability.Γ'.bit B …
                                                            -/
  ⟨⟨{.blank, .bit true, .bit false, .bra, .ket, .comma}, by decide⟩,
                                                            /-
                                                              🎉 no goals
                                                            -/
       /-
         ⊢ ∀ (x : Computability.Γ'), Membership.mem { val := Insert.insert Computabilit …
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
    by intro; cases_type* Γ' Bool <;> decide⟩
                                      /-
                                        🎉 no goals
                                      -/


instance inhabitedΓ' : Inhabited Γ' :=
  ⟨Γ'.blank⟩


/-- The natural inclusion of bool in Γ'. -/
def inclusionBoolΓ' : Bool → Γ' :=
  Γ'.bit


/-- An arbitrary section of the natural inclusion of bool in Γ'. -/
def sectionΓ'Bool : Γ' → Bool
  | Γ'.bit b => b
  | _ => Inhabited.default


theorem leftInverse_section_inclusion : Function.LeftInverse sectionΓ'Bool inclusionBoolΓ' :=
  fun x => Bool.casesOn x rfl rfl


theorem inclusionBoolΓ'_injective : Function.Injective inclusionBoolΓ' :=
  Function.HasLeftInverse.injective (Exists.intro sectionΓ'Bool leftInverse_section_inclusion)


/-- An encoding function of the positive binary numbers in bool. -/
def encodePosNum : PosNum → List Bool
  | PosNum.one    => [true]
  | PosNum.bit0 n => false :: encodePosNum n
  | PosNum.bit1 n => true :: encodePosNum n


/-- An encoding function of the binary numbers in bool. -/
def encodeNum : Num → List Bool
  | Num.zero => []
  | Num.pos n => encodePosNum n


/-- An encoding function of ℕ in bool. -/
def encodeNat (n : ℕ) : List Bool :=
  encodeNum n


/-- A decoding function from `List Bool` to the positive binary numbers. -/
def decodePosNum : List Bool → PosNum
  | false :: l => PosNum.bit0 (decodePosNum l)
  | true  :: l => ite (l = []) PosNum.one (PosNum.bit1 (decodePosNum l))
  | _          => PosNum.one


/-- A decoding function from `List Bool` to the binary numbers. -/
def decodeNum : List Bool → Num := fun l => ite (l = []) Num.zero <| decodePosNum l


/-- A decoding function from `List Bool` to ℕ. -/
def decodeNat : List Bool → Nat := fun l => decodeNum l


theorem encodePosNum_nonempty (n : PosNum) : encodePosNum n ≠ [] :=
  PosNum.casesOn n (List.cons_ne_nil _ _) (fun _m => List.cons_ne_nil _ _) fun _m =>
    List.cons_ne_nil _ _


theorem decode_encodePosNum : ∀ n, decodePosNum (encodePosNum n) = n := by
  /-
    ⊢ ∀ (n : PosNum), Eq (Computability.decodePosNum (Computability.encodePosNum n …
  -/
  intro n
  /-
    n : PosNum
    ⊢ Eq (Computability.decodePosNum (Computability.encodePosNum n)) n
  -/
  induction' n with m hm m hm <;> unfold encodePosNum decodePosNum
    /-
      case one
      ⊢ Eq (ite (Eq List.nil List.nil) PosNum.one (Computability.decodePosNum List.n …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case bit1
      m : PosNum
      hm : Eq (Computability.decodePosNum (Computability.encodePosNum m)) m
      ⊢ Eq (ite (Eq (Computability.encodePosNum m) List.nil) PosNum.one (Computabili …
    -/
  · rw [hm]
    /-
      case bit1
      m : PosNum
      hm : Eq (Computability.decodePosNum (Computability.encodePosNum m)) m
      ⊢ Eq (ite (Eq (Computability.encodePosNum m) List.nil) PosNum.one m.bit1) m.bit1
    -/
    exact if_neg (encodePosNum_nonempty m)
    /-
      🎉 no goals
    -/
    /-
      case bit0
      m : PosNum
      hm : Eq (Computability.decodePosNum (Computability.encodePosNum m)) m
      ⊢ Eq (Computability.decodePosNum (Computability.encodePosNum m)).bit0 m.bit0
    -/
  · exact congr_arg PosNum.bit0 hm
    /-
      🎉 no goals
    -/


theorem decode_encodeNum : ∀ n, decodeNum (encodeNum n) = n := by
  /-
    ⊢ ∀ (n : Num), Eq (Computability.decodeNum (Computability.encodeNum n)) n
  -/
  intro n
  /-
    n : Num
    ⊢ Eq (Computability.decodeNum (Computability.encodeNum n)) n
  -/
  cases' n with n <;> unfold encodeNum decodeNum
    /-
      case zero
      ⊢ Eq (ite (Eq (Computability.encodeNum.match_1 (fun x => List Bool) Num.zero ( …
    -/
  · rfl
    /-
      🎉 no goals
    -/
  /-
    case pos
    n : PosNum
    ⊢ Eq (ite (Eq (Computability.encodeNum.match_1 (fun x => List Bool) (Num.pos n …
  -/
  rw [decode_encodePosNum n]
  /-
    case pos
    n : PosNum
    ⊢ Eq (ite (Eq (Computability.encodeNum.match_1 (fun x => List Bool) (Num.pos n …
  -/
  rw [PosNum.cast_to_num]
  /-
    case pos
    n : PosNum
    ⊢ Eq (ite (Eq (Computability.encodeNum.match_1 (fun x => List Bool) (Num.pos n …
  -/
  exact if_neg (encodePosNum_nonempty n)
  /-
    🎉 no goals
  -/


theorem decode_encodeNat : ∀ n, decodeNat (encodeNat n) = n := by
  /-
    ⊢ ∀ (n : Nat), Eq (Computability.decodeNat (Computability.encodeNat n)) n
  -/
  intro n
  /-
    n : Nat
    ⊢ Eq (Computability.decodeNat (Computability.encodeNat n)) n
  -/
  conv_rhs => rw [← Num.to_of_nat n]
  /-
    n : Nat
    ⊢ Eq (Computability.decodeNat (Computability.encodeNat n)) ↑↑n
  -/
  exact congr_arg ((↑) : Num → ℕ) (decode_encodeNum n)
  /-
    🎉 no goals
  -/


/-- A binary encoding of ℕ in bool. -/
def encodingNatBool : Encoding ℕ where
  Γ := Bool
  encode := encodeNat
  decode n := some (decodeNat n)
  decode_encode n := congr_arg _ (decode_encodeNat n)


/-- A binary fin_encoding of ℕ in bool. -/
def finEncodingNatBool : FinEncoding ℕ :=
  ⟨encodingNatBool, Bool.fintype⟩


/-- A binary encoding of ℕ in Γ'. -/
def encodingNatΓ' : Encoding ℕ where
  Γ := Γ'
  encode x := List.map inclusionBoolΓ' (encodeNat x)
  decode x := some (decodeNat (List.map sectionΓ'Bool x))
  decode_encode x :=
    congr_arg _ <| by
      -- Porting note: `rw` can't unify `g ∘ f` with `fun x => g (f x)`, used `LeftInverse.id`
      -- instead.
      /-
        x : Nat
        ⊢ Eq (Computability.decodeNat (List.map Computability.sectionΓ'Bool ((fun x => …
      -/
      rw [List.map_map, leftInverse_section_inclusion.id, List.map_id, decode_encodeNat]
      /-
        🎉 no goals
      -/


/-- A binary fin_encoding of ℕ in Γ'. -/
def finEncodingNatΓ' : FinEncoding ℕ :=
  ⟨encodingNatΓ', Γ'.fintype⟩


/-- A unary encoding function of ℕ in bool. -/
def unaryEncodeNat : Nat → List Bool
  | 0 => []
  | n + 1 => true :: unaryEncodeNat n


/-- A unary decoding function from `List Bool` to ℕ. -/
def unaryDecodeNat : List Bool → Nat :=
  List.length


theorem unary_decode_encode_nat : ∀ n, unaryDecodeNat (unaryEncodeNat n) = n := fun n =>
  Nat.rec rfl (fun (_m : ℕ) hm => (congr_arg Nat.succ hm.symm).symm) n


/-- A unary fin_encoding of ℕ. -/
def unaryFinEncodingNat : FinEncoding ℕ where
  Γ := Bool
  encode := unaryEncodeNat
  decode n := some (unaryDecodeNat n)
  decode_encode n := congr_arg _ (unary_decode_encode_nat n)
  ΓFin := Bool.fintype


/-- An encoding function of bool in bool. -/
def encodeBool : Bool → List Bool := pure


/-- A decoding function from `List Bool` to bool. -/
def decodeBool : List Bool → Bool
  | b :: _ => b
  | _ => Inhabited.default


theorem decode_encodeBool (b : Bool) : decodeBool (encodeBool b) = b := rfl


/-- A fin_encoding of bool in bool. -/
def finEncodingBoolBool : FinEncoding Bool where
  Γ := Bool
  encode := encodeBool
  decode x := some (decodeBool x)
  decode_encode x := congr_arg _ (decode_encodeBool x)
  ΓFin := Bool.fintype


instance inhabitedFinEncoding : Inhabited (FinEncoding Bool) :=
  ⟨finEncodingBoolBool⟩


instance inhabitedEncoding : Inhabited (Encoding Bool) :=
  ⟨finEncodingBoolBool.toEncoding⟩


theorem Encoding.card_le_card_list {α : Type u} (e : Encoding.{u, v} α) :
    Cardinal.lift.{v} #α ≤ Cardinal.lift.{u} #(List e.Γ) :=
  Cardinal.lift_mk_le'.2 ⟨⟨e.encode, e.encode_injective⟩⟩


theorem Encoding.card_le_aleph0 {α : Type u} (e : Encoding.{u, v} α) [Countable e.Γ] :
    #α ≤ ℵ₀ :=
  haveI : Countable α := e.encode_injective.countable
  Cardinal.mk_le_aleph0


theorem FinEncoding.card_le_aleph0 {α : Type u} (e : FinEncoding α) : #α ≤ ℵ₀ :=
  e.toEncoding.card_le_aleph0


