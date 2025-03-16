/-- Calls the given function on a pair of entries `n`, encoded via the pairing function. -/
@[simp, reducible]
def unpaired {α} (f : ℕ → ℕ → α) (n : ℕ) : α :=
  f n.unpair.1 n.unpair.2


/-- The primitive recursive functions `ℕ → ℕ`. -/
protected inductive Primrec : (ℕ → ℕ) → Prop
  | zero : Nat.Primrec fun _ => 0
  | protected succ : Nat.Primrec succ
  | left : Nat.Primrec fun n => n.unpair.1
  | right : Nat.Primrec fun n => n.unpair.2
  | pair {f g} : Nat.Primrec f → Nat.Primrec g → Nat.Primrec fun n => pair (f n) (g n)
  | comp {f g} : Nat.Primrec f → Nat.Primrec g → Nat.Primrec fun n => f (g n)
  | prec {f g} :
      Nat.Primrec f →
        Nat.Primrec g →
          Nat.Primrec (unpaired fun z n => n.rec (f z) fun y IH => g <| pair z <| pair y IH)


theorem of_eq {f g : ℕ → ℕ} (hf : Nat.Primrec f) (H : ∀ n, f n = g n) : Nat.Primrec g :=
  (funext H : f = g) ▸ hf


theorem const : ∀ n : ℕ, Nat.Primrec fun _ => n
  | 0 => zero
  | n + 1 => Primrec.succ.comp (const n)


protected theorem id : Nat.Primrec id :=
                                      /-
                                        n : Nat
                                        ⊢ Eq (Nat.pair (Nat.unpair n).1 (Nat.unpair n).2) (id n)
                                      -/
  (left.pair right).of_eq fun n => by simp
                                      /-
                                        🎉 no goals
                                      -/


theorem prec1 {f} (m : ℕ) (hf : Nat.Primrec f) :
    Nat.Primrec fun n => n.rec m fun y IH => f <| Nat.pair y IH :=
                                                                                   /-
                                                                                     f : Nat → Nat
                                                                                     m : Nat
                                                                                     hf : Nat.Primrec f
                                                                                     n : Nat
                                                                                     ⊢ Eq (Nat.unpaired (fun z n => Nat.rec m (fun y IH => f (Nat.unpair (Nat.pair  …
                                                                                   -/
  ((prec (const m) (hf.comp right)).comp (zero.pair Primrec.id)).of_eq fun n => by simp
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


theorem casesOn1 {f} (m : ℕ) (hf : Nat.Primrec f) : Nat.Primrec (Nat.casesOn · m f) :=
                                       /-
                                         f : Nat → Nat
                                         m : Nat
                                         hf : Nat.Primrec f
                                         ⊢ ∀ (n : Nat), Eq (Nat.rec m (fun y IH => f (Nat.unpair (Nat.pair y IH)).1) n) …
                                       -/
  (prec1 m (hf.comp left)).of_eq <| by simp
                                       /-
                                         🎉 no goals
                                       -/

-- Porting note: `Nat.Primrec.casesOn` is already declared as a recursor.

theorem casesOn' {f g} (hf : Nat.Primrec f) (hg : Nat.Primrec g) :
    Nat.Primrec (unpaired fun z n => n.casesOn (f z) fun y => g <| Nat.pair z y) :=
                                                                      /-
                                                                        f g : Nat → Nat
                                                                        hf : Nat.Primrec f
                                                                        hg : Nat.Primrec g
                                                                        n : Nat
                                                                        ⊢ Eq (Nat.unpaired (fun z n => Nat.rec (f z) (fun y IH => g (Nat.pair (Nat.unp …
                                                                      -/
  (prec hf (hg.comp (pair left (left.comp right)))).of_eq fun n => by simp
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


protected theorem swap : Nat.Primrec (unpaired (swap Nat.pair)) :=
                                      /-
                                        n : Nat
                                        ⊢ Eq (Nat.pair (Nat.unpair n).2 (Nat.unpair n).1) (Nat.unpaired (Function.swap …
                                      -/
  (pair right left).of_eq fun n => by simp
                                      /-
                                        🎉 no goals
                                      -/


theorem swap' {f} (hf : Nat.Primrec (unpaired f)) : Nat.Primrec (unpaired (swap f)) :=
                                    /-
                                      f : Nat → Nat → Nat
                                      hf : Nat.Primrec (Nat.unpaired f)
                                      n : Nat
                                      ⊢ Eq (Nat.unpaired f (Nat.unpaired (Function.swap Nat.pair) n)) (Nat.unpaired  …
                                    -/
  (hf.comp .swap).of_eq fun n => by simp
                                    /-
                                      🎉 no goals
                                    -/


theorem pred : Nat.Primrec pred :=
                                            /-
                                              n : Nat
                                              ⊢ Eq (Nat.casesOn n 0 id) n.pred
                                            -/
                                                        /-
                                                          🎉 no goals
                                                        -/
  (casesOn1 0 Primrec.id).of_eq fun n => by cases n <;> simp [*]
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem add : Nat.Primrec (unpaired (· + ·)) :=
  (prec .id ((Primrec.succ.comp right).comp right)).of_eq fun p => by
    /-
      p : Nat
      ⊢ Eq (Nat.unpaired (fun z n => Nat.rec (id z) (fun y IH => (Nat.unpair (Nat.un …
    -/
                                   /-
                                     🎉 no goals
                                   -/
    simp; induction p.unpair.2 <;> simp [*, Nat.add_assoc]
                                   /-
                                     🎉 no goals
                                   -/


theorem sub : Nat.Primrec (unpaired (· - ·)) :=
  (prec .id ((pred.comp right).comp right)).of_eq fun p => by
    /-
      p : Nat
      ⊢ Eq (Nat.unpaired (fun z n => Nat.rec (id z) (fun y IH => (Nat.unpair (Nat.un …
    -/
                                   /-
                                     🎉 no goals
                                   -/
    simp; induction p.unpair.2 <;> simp [*, Nat.sub_add_eq]
                                   /-
                                     🎉 no goals
                                   -/


theorem mul : Nat.Primrec (unpaired (· * ·)) :=
  (prec zero (add.comp (pair left (right.comp right)))).of_eq fun p => by
    /-
      p : Nat
      ⊢ Eq (Nat.unpaired (fun z n => Nat.rec 0 (fun y IH => Nat.unpaired (fun x1 x2  …
    -/
                                   /-
                                     🎉 no goals
                                   -/
    simp; induction p.unpair.2 <;> simp [*, mul_succ, add_comm _ (unpair p).fst]
                                   /-
                                     🎉 no goals
                                   -/


theorem pow : Nat.Primrec (unpaired (· ^ ·)) :=
  (prec (const 1) (mul.comp (pair (right.comp right) left))).of_eq fun p => by
    /-
      p : Nat
      ⊢ Eq (Nat.unpaired (fun z n => Nat.rec 1 (fun y IH => Nat.unpaired (fun x1 x2  …
    -/
                                   /-
                                     🎉 no goals
                                   -/
    simp; induction p.unpair.2 <;> simp [*, Nat.pow_succ]
                                   /-
                                     🎉 no goals
                                   -/


/-- A `Primcodable` type is an `Encodable` type for which
  the encode/decode functions are primitive recursive. -/
class Primcodable (α : Type*) extends Encodable α where
  -- Porting note: was `prim [] `.
  -- This means that `prim` does not take the type explicitly in Lean 4
  prim : Nat.Primrec fun n => Encodable.encode (decode n)


instance (priority := 10) ofDenumerable (α) [Denumerable α] : Primcodable α :=
                                /-
                                  α : Type ?u.10160
                                  inst✝ : Denumerable α
                                  ⊢ ∀ (n : Nat), Eq n.succ (Encodable.encode (Encodable.decode n))
                                -/
  ⟨Nat.Primrec.succ.of_eq <| by simp⟩
                                /-
                                  🎉 no goals
                                -/


/-- Builds a `Primcodable` instance from an equivalence to a `Primcodable` type. -/
def ofEquiv (α) {β} [Primcodable α] (e : β ≃ α) : Primcodable β :=
  { __ := Encodable.ofEquiv α e
    prim := (@Primcodable.prim α _).of_eq fun n => by
      /-
        α : Type ?u.10721
        β : Type ?u.10728
        inst✝ : Primcodable α
        e : Equiv β α
        n : Nat
        ⊢ Eq (Encodable.encode (Encodable.decode n)) (Encodable.encode (Encodable.deco …
      -/
      rw [decode_ofEquiv]
      /-
        α : Type ?u.10721
        β : Type ?u.10728
        inst✝ : Primcodable α
        e : Equiv β α
        n : Nat
        ⊢ Eq (Encodable.encode (Encodable.decode n)) (Encodable.encode (Option.map (⇑e …
      -/
      cases (@decode α _ n) <;>
        /-
          case none
          α : Type ?u.10721
          β : Type ?u.10728
          inst✝ : Primcodable α
          e : Equiv β α
          n : Nat
          ⊢ Eq (Encodable.encode Option.none) (Encodable.encode (Option.map (⇑e.symm) Op …
        -/
        /-
          🎉 no goals
        -/
        simp [encode_ofEquiv] }
        /-
          🎉 no goals
        -/


instance empty : Primcodable Empty :=
  ⟨zero⟩


instance unit : Primcodable PUnit :=
                                       /-
                                         n : Nat
                                         ⊢ Eq (Nat.casesOn n 1 fun x => 0) (Encodable.encode (Encodable.decode n))
                                       -/
                                                   /-
                                                     🎉 no goals
                                                   -/
  ⟨(casesOn1 1 zero).of_eq fun n => by cases n <;> simp⟩
                                                   /-
                                                     🎉 no goals
                                                   -/


instance option {α : Type*} [h : Primcodable α] : Primcodable (Option α) :=
  ⟨(casesOn1 1 ((casesOn1 0 (.comp .succ .succ)).comp (@Primcodable.prim α _))).of_eq fun n => by
    cases n with
      | zero => rfl
      | succ n =>
        rw [decode_option_succ]
        cases H : @decode α _ n <;> simp [H]⟩


instance bool : Primcodable Bool :=
  ⟨(casesOn1 1 (casesOn1 2 zero)).of_eq fun n => match n with
    | 0 => rfl
    | 1 => rfl
                    /-
                      n✝ n : Nat
                      ⊢ Eq (Nat.casesOn (HAdd.hAdd n 2) 1 fun x => Nat.casesOn x 2 fun x => 0) (Enco …
                    -/
                                           /-
                                             🎉 no goals
                                           -/
    | (n + 2) => by rw [decode_ge_two] <;> simp⟩
                                           /-
                                             🎉 no goals
                                           -/


/-- `Primrec f` means `f` is primitive recursive (after
  encoding its input and output as natural numbers). -/
def Primrec {α β} [Primcodable α] [Primcodable β] (f : α → β) : Prop :=
  Nat.Primrec fun n => encode ((@decode α _ n).map f)


protected theorem encode : Primrec (@encode α _) :=
                                            /-
                                              α : Type u_1
                                              inst✝ : Primcodable α
                                              n : Nat
                                              ⊢ Eq (Encodable.encode (Encodable.decode n)) (Encodable.encode (Option.map Enc …
                                            -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  (@Primcodable.prim α _).of_eq fun n => by cases @decode α _ n <;> rfl
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


protected theorem decode : Primrec (@decode α _) :=
  Nat.Primrec.succ.comp (@Primcodable.prim α _)


theorem dom_denumerable {α β} [Denumerable α] [Primcodable β] {f : α → β} :
    Primrec f ↔ Nat.Primrec fun n => encode (f (ofNat α n)) :=
                                            /-
                                              α : Type u_4
                                              β : Type u_5
                                              inst✝¹ : Denumerable α
                                              inst✝ : Primcodable β
                                              f : α → β
                                              h : Primrec f
                                              n : Nat
                                              ⊢ Eq (Encodable.encode (Option.map f (Encodable.decode n))).pred (Encodable.en …
                                            -/
  ⟨fun h => (pred.comp h).of_eq fun n => by simp, fun h =>
                                            /-
                                              🎉 no goals
                                            -/
                                                /-
                                                  α : Type u_4
                                                  β : Type u_5
                                                  inst✝¹ : Denumerable α
                                                  inst✝ : Primcodable β
                                                  f : α → β
                                                  h : Nat.Primrec fun n => Encodable.encode (f (Denumerable.ofNat α n))
                                                  n : Nat
                                                  ⊢ Eq (Encodable.encode (f (Denumerable.ofNat α n))).succ (Encodable.encode (Op …
                                                -/
    (Nat.Primrec.succ.comp h).of_eq fun n => by simp⟩
                                                /-
                                                  🎉 no goals
                                                -/


theorem nat_iff {f : ℕ → ℕ} : Primrec f ↔ Nat.Primrec f :=
  dom_denumerable


theorem encdec : Primrec fun n => encode (@decode α _ n) :=
  nat_iff.2 Primcodable.prim


theorem option_some : Primrec (@some α) :=
  ((casesOn1 0 (Nat.Primrec.succ.comp .succ)).comp (@Primcodable.prim α _)).of_eq fun n => by
    /-
      α : Type u_1
      inst✝ : Primcodable α
      n : Nat
      ⊢ Eq (Nat.casesOn (Encodable.encode (Encodable.decode n)) 0 fun n => n.succ.su …
    -/
                            /-
                              🎉 no goals
                            -/
    cases @decode α _ n <;> simp
                            /-
                              🎉 no goals
                            -/


theorem of_eq {f g : α → σ} (hf : Primrec f) (H : ∀ n, f n = g n) : Primrec g :=
  (funext H : f = g) ▸ hf


theorem const (x : σ) : Primrec fun _ : α => x :=
  ((casesOn1 0 (.const (encode x).succ)).comp (@Primcodable.prim α _)).of_eq fun n => by
    /-
      α : Type u_1
      σ : Type u_3
      inst✝¹ : Primcodable α
      inst✝ : Primcodable σ
      x : σ
      n : Nat
      ⊢ Eq (Nat.casesOn (Encodable.encode (Encodable.decode n)) 0 fun x_1 => (Encoda …
    -/
                            /-
                              🎉 no goals
                            -/
    cases @decode α _ n <;> rfl
                            /-
                              🎉 no goals
                            -/


protected theorem id : Primrec (@id α) :=
                                    /-
                                      α : Type u_1
                                      inst✝ : Primcodable α
                                      ⊢ ∀ (n : Nat), Eq (Encodable.encode (Encodable.decode n)) (Encodable.encode (O …
                                    -/
  (@Primcodable.prim α).of_eq <| by simp
                                    /-
                                      🎉 no goals
                                    -/


theorem comp {f : β → σ} {g : α → β} (hf : Primrec f) (hg : Primrec g) : Primrec fun a => f (g a) :=
  ((casesOn1 0 (.comp hf (pred.comp hg))).comp (@Primcodable.prim α _)).of_eq fun n => by
    /-
      α : Type u_1
      β : Type u_2
      σ : Type u_3
      inst✝² : Primcodable α
      inst✝¹ : Primcodable β
      inst✝ : Primcodable σ
      f : β → σ
      g : α → β
      hf : Primrec f
      hg : Primrec g
      n : Nat
      ⊢ Eq (Nat.casesOn (Encodable.encode (Encodable.decode n)) 0 fun n => Encodable …
    -/
                            /-
                              🎉 no goals
                            -/
    cases @decode α _ n <;> simp [encodek]
                            /-
                              🎉 no goals
                            -/


theorem succ : Primrec Nat.succ :=
  nat_iff.2 Nat.Primrec.succ


theorem pred : Primrec Nat.pred :=
  nat_iff.2 Nat.Primrec.pred


theorem encode_iff {f : α → σ} : (Primrec fun a => encode (f a)) ↔ Primrec f :=
                                            /-
                                              α : Type u_1
                                              σ : Type u_3
                                              inst✝¹ : Primcodable α
                                              inst✝ : Primcodable σ
                                              f : α → σ
                                              h : Primrec fun a => Encodable.encode (f a)
                                              n : Nat
                                              ⊢ Eq (Encodable.encode (Option.map (fun a => Encodable.encode (f a)) (Encodabl …
                                            -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  ⟨fun h => Nat.Primrec.of_eq h fun n => by cases @decode α _ n <;> rfl, Primrec.encode.comp⟩
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem ofNat_iff {α β} [Denumerable α] [Primcodable β] {f : α → β} :
    Primrec f ↔ Primrec fun n => f (ofNat α n) :=
  dom_denumerable.trans <| nat_iff.symm.trans encode_iff


protected theorem ofNat (α) [Denumerable α] : Primrec (ofNat α) :=
  ofNat_iff.1 Primrec.id


theorem option_some_iff {f : α → σ} : (Primrec fun a => some (f a)) ↔ Primrec f :=
  ⟨fun h => encode_iff.1 <| pred.comp <| encode_iff.2 h, option_some.comp⟩


theorem of_equiv {β} {e : β ≃ α} :
    haveI := Primcodable.ofEquiv α e
    Primrec e :=
  letI : Primcodable β := Primcodable.ofEquiv α e
  encode_iff.1 Primrec.encode


theorem of_equiv_symm {β} {e : β ≃ α} :
    haveI := Primcodable.ofEquiv α e
    Primrec e.symm :=
  letI := Primcodable.ofEquiv α e
                                                               /-
                                                                 α : Type u_1
                                                                 inst✝ : Primcodable α
                                                                 β : Type u_4
                                                                 e : Equiv β α
                                                                 this : Primcodable β := Primcodable.ofEquiv α e
                                                                 ⊢ Primrec fun a => Encodable.encode (e (e.symm a))
                                                               -/
  encode_iff.1 (show Primrec fun a => encode (e (e.symm a)) by simp [Primrec.encode])
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem of_equiv_iff {β} (e : β ≃ α) {f : σ → β} :
    haveI := Primcodable.ofEquiv α e
    (Primrec fun a => e (f a)) ↔ Primrec f :=
  letI := Primcodable.ofEquiv α e
                                                     /-
                                                       α : Type u_1
                                                       σ : Type u_3
                                                       inst✝¹ : Primcodable α
                                                       inst✝ : Primcodable σ
                                                       β : Type u_4
                                                       e : Equiv β α
                                                       f : σ → β
                                                       this : Primcodable β := Primcodable.ofEquiv α e
                                                       h : Primrec fun a => e (f a)
                                                       a : σ
                                                       ⊢ Eq (e.symm (e (f a))) (f a)
                                                     -/
  ⟨fun h => (of_equiv_symm.comp h).of_eq fun a => by simp, of_equiv.comp⟩
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem of_equiv_symm_iff {β} (e : β ≃ α) {f : σ → α} :
    haveI := Primcodable.ofEquiv α e
    (Primrec fun a => e.symm (f a)) ↔ Primrec f :=
  letI := Primcodable.ofEquiv α e
                                                /-
                                                  α : Type u_1
                                                  σ : Type u_3
                                                  inst✝¹ : Primcodable α
                                                  inst✝ : Primcodable σ
                                                  β : Type u_4
                                                  e : Equiv β α
                                                  f : σ → α
                                                  this : Primcodable β := Primcodable.ofEquiv α e
                                                  h : Primrec fun a => e.symm (f a)
                                                  a : σ
                                                  ⊢ Eq (e (e.symm (f a))) (f a)
                                                -/
  ⟨fun h => (of_equiv.comp h).of_eq fun a => by simp, of_equiv_symm.comp⟩
                                                /-
                                                  🎉 no goals
                                                -/


instance prod {α β} [Primcodable α] [Primcodable β] : Primcodable (α × β) :=
  ⟨((casesOn' zero ((casesOn' zero .succ).comp (pair right ((@Primcodable.prim β).comp left)))).comp
          (pair right ((@Primcodable.prim α).comp left))).of_eq
      fun n => by
      /-
        α : Type ?u.24224
        β : Type ?u.24227
        inst✝¹ : Primcodable α
        inst✝ : Primcodable β
        n : Nat
        ⊢ Eq (Nat.unpaired (fun z n => Nat.casesOn n 0 fun y => Nat.unpaired (fun z n  …
      -/
      simp only [Nat.unpaired, Nat.unpair_pair, decode_prod_val]
      /-
        α : Type ?u.24224
        β : Type ?u.24227
        inst✝¹ : Primcodable α
        inst✝ : Primcodable β
        n : Nat
        ⊢ Eq (Nat.rec 0 (fun n_1 n_ih => Nat.rec 0 (fun n n_ih => (Nat.pair n_1 n).suc …
      -/
      cases @decode α _ n.unpair.1; · simp
                                      /-
                                        🎉 no goals
                                      -/
      /-
        case some
        α : Type ?u.24224
        β : Type ?u.24227
        inst✝¹ : Primcodable α
        inst✝ : Primcodable β
        n : Nat
        val✝ : α
        ⊢ Eq (Nat.rec 0 (fun n_1 n_ih => Nat.rec 0 (fun n n_ih => (Nat.pair n_1 n).suc …
      -/
                                       /-
                                         🎉 no goals
                                       -/
      cases @decode β _ n.unpair.2 <;> simp⟩
                                       /-
                                         🎉 no goals
                                       -/


theorem fst {α β} [Primcodable α] [Primcodable β] : Primrec (@Prod.fst α β) :=
  ((casesOn' zero
            ((casesOn' zero (Nat.Primrec.succ.comp left)).comp
              (pair right ((@Primcodable.prim β).comp left)))).comp
        (pair right ((@Primcodable.prim α).comp left))).of_eq
    fun n => by
    /-
      α : Type u_2
      β : Type u_3
      inst✝¹ : Primcodable α
      inst✝ : Primcodable β
      n : Nat
      ⊢ Eq (Nat.unpaired (fun z n => Nat.casesOn n 0 fun y => Nat.unpaired (fun z n  …
    -/
    simp only [Nat.unpaired, Nat.unpair_pair, decode_prod_val]
    /-
      α : Type u_2
      β : Type u_3
      inst✝¹ : Primcodable α
      inst✝ : Primcodable β
      n : Nat
      ⊢ Eq (Nat.rec 0 (fun n_1 n_ih => Nat.rec 0 (fun n n_ih => n_1.succ) (Encodable …
    -/
                                     /-
                                       🎉 no goals
                                     -/
    cases @decode α _ n.unpair.1 <;> simp
    /-
      case some
      α : Type u_2
      β : Type u_3
      inst✝¹ : Primcodable α
      inst✝ : Primcodable β
      n : Nat
      val✝ : α
      ⊢ Eq (Nat.rec 0 (fun n n_ih => HAdd.hAdd (Encodable.encode val✝) 1) (Encodable …
    -/
                                     /-
                                       🎉 no goals
                                     -/
    cases @decode β _ n.unpair.2 <;> simp
                                     /-
                                       🎉 no goals
                                     -/


theorem snd {α β} [Primcodable α] [Primcodable β] : Primrec (@Prod.snd α β) :=
  ((casesOn' zero
            ((casesOn' zero (Nat.Primrec.succ.comp right)).comp
              (pair right ((@Primcodable.prim β).comp left)))).comp
        (pair right ((@Primcodable.prim α).comp left))).of_eq
    fun n => by
    /-
      α : Type u_2
      β : Type u_3
      inst✝¹ : Primcodable α
      inst✝ : Primcodable β
      n : Nat
      ⊢ Eq (Nat.unpaired (fun z n => Nat.casesOn n 0 fun y => Nat.unpaired (fun z n  …
    -/
    simp only [Nat.unpaired, Nat.unpair_pair, decode_prod_val]
    /-
      α : Type u_2
      β : Type u_3
      inst✝¹ : Primcodable α
      inst✝ : Primcodable β
      n : Nat
      ⊢ Eq (Nat.rec 0 (fun n_1 n_ih => Nat.rec 0 (fun n n_ih => n.succ) (Encodable.e …
    -/
                                     /-
                                       🎉 no goals
                                     -/
    cases @decode α _ n.unpair.1 <;> simp
    /-
      case some
      α : Type u_2
      β : Type u_3
      inst✝¹ : Primcodable α
      inst✝ : Primcodable β
      n : Nat
      val✝ : α
      ⊢ Eq (Nat.rec 0 (fun n n_ih => HAdd.hAdd n 1) (Encodable.encode (Encodable.dec …
    -/
                                     /-
                                       🎉 no goals
                                     -/
    cases @decode β _ n.unpair.2 <;> simp
                                     /-
                                       🎉 no goals
                                     -/


theorem pair {α β γ} [Primcodable α] [Primcodable β] [Primcodable γ] {f : α → β} {g : α → γ}
    (hf : Primrec f) (hg : Primrec g) : Primrec fun a => (f a, g a) :=
  ((casesOn1 0
            (Nat.Primrec.succ.comp <|
              .pair (Nat.Primrec.pred.comp hf) (Nat.Primrec.pred.comp hg))).comp
        (@Primcodable.prim α _)).of_eq
                /-
                  α : Type u_2
                  β : Type u_3
                  γ : Type u_4
                  inst✝² : Primcodable α
                  inst✝¹ : Primcodable β
                  inst✝ : Primcodable γ
                  f : α → β
                  g : α → γ
                  hf : Primrec f
                  hg : Primrec g
                  n : Nat
                  ⊢ Eq (Nat.casesOn (Encodable.encode (Encodable.decode n)) 0 fun n => (Nat.pair …
                -/
                                        /-
                                          🎉 no goals
                                        -/
    fun n => by cases @decode α _ n <;> simp [encodek]
                                        /-
                                          🎉 no goals
                                        -/


theorem unpair : Primrec Nat.unpair :=
                                                                /-
                                                                  n : Nat
                                                                  ⊢ Eq { fst := (Nat.unpair n).1, snd := (Nat.unpair n).2 } (Nat.unpair n)
                                                                -/
  (pair (nat_iff.2 .left) (nat_iff.2 .right)).of_eq fun n => by simp
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem list_get?₁ : ∀ l : List α, Primrec l.get?
  | [] => dom_denumerable.2 zero
  | a :: l =>
    dom_denumerable.2 <|
      (casesOn1 (encode a).succ <| dom_denumerable.1 <| list_get?₁ l).of_eq fun n => by
        /-
          α : Type u_1
          inst✝ : Primcodable α
          a : α
          l : List α
          n : Nat
          ⊢ Eq (Nat.casesOn n (Encodable.encode a).succ fun n => Encodable.encode (l.get …
        -/
                    /-
                      🎉 no goals
                    -/
        cases n <;> simp
                    /-
                      🎉 no goals
                    -/


/-- `Primrec₂ f` means `f` is a binary primitive recursive function.
  This is technically unnecessary since we can always curry all
  the arguments together, but there are enough natural two-arg
  functions that it is convenient to express this directly. -/
def Primrec₂ {α β σ} [Primcodable α] [Primcodable β] [Primcodable σ] (f : α → β → σ) :=
  Primrec fun p : α × β => f p.1 p.2


/-- `PrimrecPred p` means `p : α → Prop` is a (decidable)
  primitive recursive predicate, which is to say that
  `decide ∘ p : α → Bool` is primitive recursive. -/
def PrimrecPred {α} [Primcodable α] (p : α → Prop) [DecidablePred p] :=
  Primrec fun a => decide (p a)


/-- `PrimrecRel p` means `p : α → β → Prop` is a (decidable)
  primitive recursive relation, which is to say that
  `decide ∘ p : α → β → Bool` is primitive recursive. -/
def PrimrecRel {α β} [Primcodable α] [Primcodable β] (s : α → β → Prop)
    [∀ a b, Decidable (s a b)] :=
  Primrec₂ fun a b => decide (s a b)


theorem mk {f : α → β → σ} (hf : Primrec fun p : α × β => f p.1 p.2) : Primrec₂ f := hf


theorem of_eq {f g : α → β → σ} (hg : Primrec₂ f) (H : ∀ a b, f a b = g a b) : Primrec₂ g :=
      /-
        α : Type u_1
        β : Type u_2
        σ : Type u_3
        inst✝² : Primcodable α
        inst✝¹ : Primcodable β
        inst✝ : Primcodable σ
        f g : α → β → σ
        hg : Primrec₂ f
        H : ∀ (a : α) (b : β), Eq (f a b) (g a b)
        ⊢ Eq f g
      -/
  (by funext a b; apply H : f = g) ▸ hg
                  /-
                    🎉 no goals
                  -/


theorem const (x : σ) : Primrec₂ fun (_ : α) (_ : β) => x :=
  Primrec.const _


protected theorem pair : Primrec₂ (@Prod.mk α β) :=
  .pair .fst .snd


theorem left : Primrec₂ fun (a : α) (_ : β) => a :=
  .fst


theorem right : Primrec₂ fun (_ : α) (b : β) => b :=
  .snd


                                          /-
                                            ⊢ Primrec₂ Nat.pair
                                          -/
theorem natPair : Primrec₂ Nat.pair := by simp [Primrec₂, Primrec]; constructor
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem unpaired {f : ℕ → ℕ → α} : Primrec (Nat.unpaired f) ↔ Primrec₂ f :=
               /-
                 α : Type u_1
                 inst✝ : Primcodable α
                 f : Nat → Nat → α
                 h : Primrec (Nat.unpaired f)
                 ⊢ Primrec₂ f
               -/
  ⟨fun h => by simpa using h.comp natPair, fun h => h.comp Primrec.unpair⟩
               /-
                 🎉 no goals
               -/


theorem unpaired' {f : ℕ → ℕ → ℕ} : Nat.Primrec (Nat.unpaired f) ↔ Primrec₂ f :=
  Primrec.nat_iff.symm.trans unpaired


theorem encode_iff {f : α → β → σ} : (Primrec₂ fun a b => encode (f a b)) ↔ Primrec₂ f :=
  Primrec.encode_iff


theorem option_some_iff {f : α → β → σ} : (Primrec₂ fun a b => some (f a b)) ↔ Primrec₂ f :=
  Primrec.option_some_iff


theorem ofNat_iff {α β σ} [Denumerable α] [Denumerable β] [Primcodable σ] {f : α → β → σ} :
    Primrec₂ f ↔ Primrec₂ fun m n : ℕ => f (ofNat α m) (ofNat β n) :=
                                 /-
                                   α : Type u_4
                                   β : Type u_5
                                   σ : Type u_6
                                   inst✝² : Denumerable α
                                   inst✝¹ : Denumerable β
                                   inst✝ : Primcodable σ
                                   f : α → β → σ
                                   ⊢ Iff (Primrec fun n => f (Denumerable.ofNat (Prod α β) n).1 (Denumerable.ofNa …
                                 -/
  (Primrec.ofNat_iff.trans <| by simp).trans unpaired
                                 /-
                                   🎉 no goals
                                 -/


theorem uncurry {f : α → β → σ} : Primrec (Function.uncurry f) ↔ Primrec₂ f := by
  /-
    α : Type u_1
    β : Type u_2
    σ : Type u_3
    inst✝² : Primcodable α
    inst✝¹ : Primcodable β
    inst✝ : Primcodable σ
    f : α → β → σ
    ⊢ Iff (Primrec (Function.uncurry f)) (Primrec₂ f)
  -/
  rw [show Function.uncurry f = fun p : α × β => f p.1 p.2 from funext fun ⟨a, b⟩ => rfl]; rfl
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/


theorem curry {f : α × β → σ} : Primrec₂ (Function.curry f) ↔ Primrec f := by
  /-
    α : Type u_1
    β : Type u_2
    σ : Type u_3
    inst✝² : Primcodable α
    inst✝¹ : Primcodable β
    inst✝ : Primcodable σ
    f : Prod α β → σ
    ⊢ Iff (Primrec₂ (Function.curry f)) (Primrec f)
  -/
  rw [← uncurry, Function.uncurry_curry]
  /-
    🎉 no goals
  -/


theorem Primrec.comp₂ {f : γ → σ} {g : α → β → γ} (hf : Primrec f) (hg : Primrec₂ g) :
    Primrec₂ fun a b => f (g a b) :=
  hf.comp hg


theorem Primrec₂.comp {f : β → γ → σ} {g : α → β} {h : α → γ} (hf : Primrec₂ f) (hg : Primrec g)
    (hh : Primrec h) : Primrec fun a => f (g a) (h a) :=
  Primrec.comp hf (hg.pair hh)


theorem Primrec₂.comp₂ {f : γ → δ → σ} {g : α → β → γ} {h : α → β → δ} (hf : Primrec₂ f)
    (hg : Primrec₂ g) (hh : Primrec₂ h) : Primrec₂ fun a b => f (g a b) (h a b) :=
  hf.comp hg hh


theorem PrimrecPred.comp {p : β → Prop} [DecidablePred p] {f : α → β} :
    PrimrecPred p → Primrec f → PrimrecPred fun a => p (f a) :=
  Primrec.comp


theorem PrimrecRel.comp {R : β → γ → Prop} [∀ a b, Decidable (R a b)] {f : α → β} {g : α → γ} :
    PrimrecRel R → Primrec f → Primrec g → PrimrecPred fun a => R (f a) (g a) :=
  Primrec₂.comp


theorem PrimrecRel.comp₂ {R : γ → δ → Prop} [∀ a b, Decidable (R a b)] {f : α → β → γ}
    {g : α → β → δ} :
    PrimrecRel R → Primrec₂ f → Primrec₂ g → PrimrecRel fun a b => R (f a b) (g a b) :=
  PrimrecRel.comp


theorem PrimrecPred.of_eq {α} [Primcodable α] {p q : α → Prop} [DecidablePred p] [DecidablePred q]
    (hp : PrimrecPred p) (H : ∀ a, p a ↔ q a) : PrimrecPred q :=
  Primrec.of_eq hp fun a => Bool.decide_congr (H a)


theorem PrimrecRel.of_eq {α β} [Primcodable α] [Primcodable β] {r s : α → β → Prop}
    [∀ a b, Decidable (r a b)] [∀ a b, Decidable (s a b)] (hr : PrimrecRel r)
    (H : ∀ a b, r a b ↔ s a b) : PrimrecRel s :=
  Primrec₂.of_eq hr fun a b => Bool.decide_congr (H a b)


theorem swap {f : α → β → σ} (h : Primrec₂ f) : Primrec₂ (swap f) :=
  h.comp₂ Primrec₂.right Primrec₂.left


theorem nat_iff {f : α → β → σ} : Primrec₂ f ↔ Nat.Primrec
    (.unpaired fun m n => encode <| (@decode α _ m).bind fun a => (@decode β _ n).map (f a)) := by
  have :
    ∀ (a : Option α) (b : Option β),
      Option.map (fun p : α × β => f p.1 p.2)
          (Option.bind a fun a : α => Option.map (Prod.mk a) b) =
        Option.bind a fun a => Option.map (f a) b := fun a b => by
          cases a <;> cases b <;> rfl
  /-
    α : Type u_1
    β : Type u_2
    σ : Type u_3
    inst✝² : Primcodable α
    inst✝¹ : Primcodable β
    inst✝ : Primcodable σ
    f : α → β → σ
    this : ∀ (a : Option α) (b : Option β), Eq (Option.map (fun p => f p.1 p.2) (a …
    ⊢ Iff (Primrec₂ f) (Nat.Primrec (Nat.unpaired fun m n => Encodable.encode ((En …
  -/
  simp [Primrec₂, Primrec, this]
  /-
    🎉 no goals
  -/


theorem nat_iff' {f : α → β → σ} :
    Primrec₂ f ↔
      Primrec₂ fun m n : ℕ => (@decode α _ m).bind fun a => Option.map (f a) (@decode β _ n) :=
  nat_iff.trans <| unpaired'.trans encode_iff


theorem to₂ {f : α × β → σ} (hf : Primrec f) : Primrec₂ fun a b => f (a, b) :=
  hf.of_eq fun _ => rfl


theorem nat_rec {f : α → β} {g : α → ℕ × β → β} (hf : Primrec f) (hg : Primrec₂ g) :
    Primrec₂ fun a (n : ℕ) => n.rec (motive := fun _ => β) (f a) fun n IH => g a (n, IH) :=
  Primrec₂.nat_iff.2 <|
    ((Nat.Primrec.casesOn' .zero <|
              (Nat.Primrec.prec hf <|
                    .comp hg <|
                      Nat.Primrec.left.pair <|
                        (Nat.Primrec.left.comp .right).pair <|
                          Nat.Primrec.pred.comp <| Nat.Primrec.right.comp .right).comp <|
                Nat.Primrec.right.pair <| Nat.Primrec.right.comp Nat.Primrec.left).comp <|
          Nat.Primrec.id.pair <| (@Primcodable.prim α).comp Nat.Primrec.left).of_eq
      fun n => by
      simp only [Nat.unpaired, id_eq, Nat.unpair_pair, decode_prod_val, decode_nat,
        Option.some_bind, Option.map_map, Option.map_some']
      /-
        α : Type u_1
        β : Type u_2
        inst✝¹ : Primcodable α
        inst✝ : Primcodable β
        f : α → β
        g : α → Prod Nat β → β
        hf : Primrec f
        hg : Primrec₂ g
        n : Nat
        ⊢ Eq (Nat.rec 0 (fun n_1 n_ih => Nat.rec (Encodable.encode (Option.map f (Enco …
      -/
      cases' @decode α _ n.unpair.1 with a; · rfl
                                              /-
                                                🎉 no goals
                                              -/
      simp only [Nat.pred_eq_sub_one, encode_some, Nat.succ_eq_add_one, encodek, Option.map_some',
        Option.some_bind, Option.map_map]
      /-
        case some
        α : Type u_1
        β : Type u_2
        inst✝¹ : Primcodable α
        inst✝ : Primcodable β
        f : α → β
        g : α → Prod Nat β → β
        hf : Primrec f
        hg : Primrec₂ g
        n : Nat
        a : α
        ⊢ Eq (Nat.rec (HAdd.hAdd (Encodable.encode (f a)) 1) (fun y IH => Encodable.en …
      -/
                                       /-
                                         🎉 no goals
                                       -/
      induction' n.unpair.2 with m <;> simp [encodek]
      /-
        case some.succ
        α : Type u_1
        β : Type u_2
        inst✝¹ : Primcodable α
        inst✝ : Primcodable β
        f : α → β
        g : α → Prod Nat β → β
        hf : Primrec f
        hg : Primrec₂ g
        n : Nat
        a : α
        m : Nat
        a✝ : Eq (Nat.rec (HAdd.hAdd (Encodable.encode (f a)) 1) (fun y IH => Encodable …
        ⊢ Eq (Encodable.encode (Option.map (Function.comp (fun p => g p.1 p.2) (Functi …
      -/
      simp [*, encodek]
      /-
        🎉 no goals
      -/


theorem nat_rec' {f : α → ℕ} {g : α → β} {h : α → ℕ × β → β}
    (hf : Primrec f) (hg : Primrec g) (hh : Primrec₂ h) :
    Primrec fun a => (f a).rec (motive := fun _ => β) (g a) fun n IH => h a (n, IH) :=
  (nat_rec hg hh).comp .id hf


theorem nat_rec₁ {f : ℕ → α → α} (a : α) (hf : Primrec₂ f) : Primrec (Nat.rec a f) :=
  nat_rec' .id (const a) <| comp₂ hf Primrec₂.right


theorem nat_casesOn' {f : α → β} {g : α → ℕ → β} (hf : Primrec f) (hg : Primrec₂ g) :
    Primrec₂ fun a (n : ℕ) => (n.casesOn (f a) (g a) : β) :=
  nat_rec hf <| hg.comp₂ Primrec₂.left <| comp₂ fst Primrec₂.right


theorem nat_casesOn {f : α → ℕ} {g : α → β} {h : α → ℕ → β} (hf : Primrec f) (hg : Primrec g)
    (hh : Primrec₂ h) : Primrec fun a => ((f a).casesOn (g a) (h a) : β) :=
  (nat_casesOn' hg hh).comp .id hf


theorem nat_casesOn₁ {f : ℕ → α} (a : α) (hf : Primrec f) :
    Primrec (fun (n : ℕ) => (n.casesOn a f : α)) :=
  nat_casesOn .id (const a) (comp₂ hf .right)


theorem nat_iterate {f : α → ℕ} {g : α → β} {h : α → β → β} (hf : Primrec f) (hg : Primrec g)
    (hh : Primrec₂ h) : Primrec fun a => (h a)^[f a] (g a) :=
  (nat_rec' hf hg (hh.comp₂ Primrec₂.left <| snd.comp₂ Primrec₂.right)).of_eq fun a => by
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : Primcodable α
      inst✝ : Primcodable β
      f : α → Nat
      g : α → β
      h : α → β → β
      hf : Primrec f
      hg : Primrec g
      hh : Primrec₂ h
      a : α
      ⊢ Eq (Nat.rec (g a) (fun n IH => h a { fst := n, snd := IH }.2) (f a)) (Nat.it …
    -/
                      /-
                        🎉 no goals
                      -/
    induction f a <;> simp [*, -Function.iterate_succ, Function.iterate_succ']
                      /-
                        🎉 no goals
                      -/


theorem option_casesOn {o : α → Option β} {f : α → σ} {g : α → β → σ} (ho : Primrec o)
    (hf : Primrec f) (hg : Primrec₂ g) :
    @Primrec _ σ _ _ fun a => Option.casesOn (o a) (f a) (g a) :=
  encode_iff.1 <|
    (nat_casesOn (encode_iff.2 ho) (encode_iff.2 hf) <|
          pred.comp₂ <|
            Primrec₂.encode_iff.2 <|
              (Primrec₂.nat_iff'.1 hg).comp₂ ((@Primrec.encode α _).comp fst).to₂
                Primrec₂.right).of_eq
                  /-
                    α : Type u_1
                    β : Type u_2
                    σ : Type u_3
                    inst✝² : Primcodable α
                    inst✝¹ : Primcodable β
                    inst✝ : Primcodable σ
                    o : α → Option β
                    f : α → σ
                    g : α → β → σ
                    ho : Primrec o
                    hf : Primrec f
                    hg : Primrec₂ g
                    a : α
                    ⊢ Eq (Nat.casesOn (Encodable.encode (o a)) (Encodable.encode (f a)) fun b => ( …
                  -/
                                        /-
                                          🎉 no goals
                                        -/
      fun a => by cases' o a with b <;> simp [encodek]
                                        /-
                                          🎉 no goals
                                        -/


theorem option_bind {f : α → Option β} {g : α → β → Option σ} (hf : Primrec f) (hg : Primrec₂ g) :
    Primrec fun a => (f a).bind (g a) :=
                                                        /-
                                                          α : Type u_1
                                                          β : Type u_2
                                                          σ : Type u_3
                                                          inst✝² : Primcodable α
                                                          inst✝¹ : Primcodable β
                                                          inst✝ : Primcodable σ
                                                          f : α → Option β
                                                          g : α → β → Option σ
                                                          hf : Primrec f
                                                          hg : Primrec₂ g
                                                          a : α
                                                          ⊢ Eq (Option.casesOn (f a) Option.none (g a)) ((f a).bind (g a))
                                                        -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
  (option_casesOn hf (const none) hg).of_eq fun a => by cases f a <;> rfl
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem option_bind₁ {f : α → Option σ} (hf : Primrec f) : Primrec fun o => Option.bind o f :=
  option_bind .id (hf.comp snd).to₂


theorem option_map {f : α → Option β} {g : α → β → σ} (hf : Primrec f) (hg : Primrec₂ g) :
    Primrec fun a => (f a).map (g a) :=
                                                            /-
                                                              α : Type u_1
                                                              β : Type u_2
                                                              σ : Type u_3
                                                              inst✝² : Primcodable α
                                                              inst✝¹ : Primcodable β
                                                              inst✝ : Primcodable σ
                                                              f : α → Option β
                                                              g : α → β → σ
                                                              hf : Primrec f
                                                              hg : Primrec₂ g
                                                              x : α
                                                              ⊢ Eq ((f x).bind fun b => Option.some (g x b)) (Option.map (g x) (f x))
                                                            -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
  (option_bind hf (option_some.comp₂ hg)).of_eq fun x => by cases f x <;> rfl
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


theorem option_map₁ {f : α → σ} (hf : Primrec f) : Primrec (Option.map f) :=
  option_map .id (hf.comp snd).to₂


theorem option_iget [Inhabited α] : Primrec (@Option.iget α _) :=
                                                                        /-
                                                                          α : Type u_1
                                                                          inst✝¹ : Primcodable α
                                                                          inst✝ : Inhabited α
                                                                          o : Option α
                                                                          ⊢ Eq (Option.casesOn (id o) Inhabited.default fun b => b) o.iget
                                                                        -/
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
  (option_casesOn .id (const <| @default α _) .right).of_eq fun o => by cases o <;> rfl
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


theorem option_isSome : Primrec (@Option.isSome α) :=
                                                                        /-
                                                                          α : Type u_1
                                                                          inst✝ : Primcodable α
                                                                          o : Option α
                                                                          ⊢ Eq (Option.casesOn (id o) Bool.false fun b => Bool.true) o.isSome
                                                                        -/
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
  (option_casesOn .id (const false) (const true).to₂).of_eq fun o => by cases o <;> rfl
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


theorem option_getD : Primrec₂ (@Option.getD α) :=
  Primrec.of_eq (option_casesOn Primrec₂.left Primrec₂.right .right) fun ⟨o, a⟩ => by
    /-
      α : Type u_1
      inst✝ : Primcodable α
      x✝ : Prod (Option α) α
      o : Option α
      a : α
      ⊢ Eq (Option.casesOn ((fun a x => a) { fst := o, snd := a }.1 { fst := o, snd  …
    -/
                /-
                  🎉 no goals
                -/
    cases o <;> rfl
                /-
                  🎉 no goals
                -/


theorem bind_decode_iff {f : α → β → Option σ} :
    (Primrec₂ fun a n => (@decode β _ n).bind (f a)) ↔ Primrec₂ f :=
               /-
                 α : Type u_1
                 β : Type u_2
                 σ : Type u_3
                 inst✝² : Primcodable α
                 inst✝¹ : Primcodable β
                 inst✝ : Primcodable σ
                 f : α → β → Option σ
                 h : Primrec₂ fun a n => (Encodable.decode n).bind (f a)
                 ⊢ Primrec₂ f
               -/
  ⟨fun h => by simpa [encodek] using h.comp fst ((@Primrec.encode β _).comp snd), fun h =>
               /-
                 🎉 no goals
               -/
    option_bind (Primrec.decode.comp snd) <| h.comp (fst.comp fst) snd⟩


theorem map_decode_iff {f : α → β → σ} :
    (Primrec₂ fun a n => (@decode β _ n).map (f a)) ↔ Primrec₂ f := by
  /-
    α : Type u_1
    β : Type u_2
    σ : Type u_3
    inst✝² : Primcodable α
    inst✝¹ : Primcodable β
    inst✝ : Primcodable σ
    f : α → β → σ
    ⊢ Iff (Primrec₂ fun a n => Option.map (f a) (Encodable.decode n)) (Primrec₂ f)
  -/
  simp only [Option.map_eq_bind]
  /-
    α : Type u_1
    β : Type u_2
    σ : Type u_3
    inst✝² : Primcodable α
    inst✝¹ : Primcodable β
    inst✝ : Primcodable σ
    f : α → β → σ
    ⊢ Iff (Primrec₂ fun a n => (Encodable.decode n).bind (Function.comp Option.som …
  -/
  exact bind_decode_iff.trans Primrec₂.option_some_iff
  /-
    🎉 no goals
  -/


theorem nat_add : Primrec₂ ((· + ·) : ℕ → ℕ → ℕ) :=
  Primrec₂.unpaired'.1 Nat.Primrec.add


theorem nat_sub : Primrec₂ ((· - ·) : ℕ → ℕ → ℕ) :=
  Primrec₂.unpaired'.1 Nat.Primrec.sub


theorem nat_mul : Primrec₂ ((· * ·) : ℕ → ℕ → ℕ) :=
  Primrec₂.unpaired'.1 Nat.Primrec.mul


theorem cond {c : α → Bool} {f : α → σ} {g : α → σ} (hc : Primrec c) (hf : Primrec f)
    (hg : Primrec g) : Primrec fun a => bif (c a) then (f a) else (g a) :=
                                                                         /-
                                                                           α : Type u_1
                                                                           σ : Type u_3
                                                                           inst✝¹ : Primcodable α
                                                                           inst✝ : Primcodable σ
                                                                           c : α → Bool
                                                                           f g : α → σ
                                                                           hc : Primrec c
                                                                           hf : Primrec f
                                                                           hg : Primrec g
                                                                           a : α
                                                                           ⊢ Eq (Nat.casesOn (Encodable.encode (c a)) (g a) fun b => f { fst := a, snd := …
                                                                         -/
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/
  (nat_casesOn (encode_iff.2 hc) hg (hf.comp fst).to₂).of_eq fun a => by cases c a <;> rfl
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


theorem ite {c : α → Prop} [DecidablePred c] {f : α → σ} {g : α → σ} (hc : PrimrecPred c)
    (hf : Primrec f) (hg : Primrec g) : Primrec fun a => if c a then f a else g a := by
  /-
    α : Type u_1
    σ : Type u_3
    inst✝² : Primcodable α
    inst✝¹ : Primcodable σ
    c : α → Prop
    inst✝ : DecidablePred c
    f g : α → σ
    hc : PrimrecPred c
    hf : Primrec f
    hg : Primrec g
    ⊢ Primrec fun a => _root_.ite (c a) (f a) (g a)
  -/
  simpa [Bool.cond_decide] using cond hc hf hg
  /-
    🎉 no goals
  -/


theorem nat_le : PrimrecRel ((· ≤ ·) : ℕ → ℕ → Prop) :=
  (nat_casesOn nat_sub (const true) (const false).to₂).of_eq fun p => by
    /-
      p : Prod Nat Nat
      ⊢ Eq (Nat.casesOn ((fun x1 x2 => HSub.hSub x1 x2) p.1 p.2) Bool.true fun b =>  …
    -/
    dsimp [swap]
    /-
      p : Prod Nat Nat
      ⊢ Eq (Nat.rec Bool.true (fun n n_ih => Bool.false) (HSub.hSub p.1 p.2)) (Decid …
    -/
    cases' e : p.1 - p.2 with n
      /-
        case zero
        p : Prod Nat Nat
        e : Eq (HSub.hSub p.1 p.2) 0
        ⊢ Eq (Nat.rec Bool.true (fun n n_ih => Bool.false) 0) (Decidable.decide (LE.le …
      -/
    · simp [tsub_eq_zero_iff_le.1 e]
      /-
        🎉 no goals
      -/
      /-
        case succ
        p : Prod Nat Nat
        n : Nat
        e : Eq (HSub.hSub p.1 p.2) (HAdd.hAdd n 1)
        ⊢ Eq (Nat.rec Bool.true (fun n n_ih => Bool.false) (HAdd.hAdd n 1)) (Decidable …
      -/
    · simp [not_le.2 (Nat.lt_of_sub_eq_succ e)]
      /-
        🎉 no goals
      -/


theorem nat_min : Primrec₂ (@min ℕ _) :=
  ite nat_le fst snd


theorem nat_max : Primrec₂ (@max ℕ _) :=
  ite (nat_le.comp fst snd) snd fst


theorem dom_bool (f : Bool → α) : Primrec f :=
                                                                  /-
                                                                    α : Type u_1
                                                                    inst✝ : Primcodable α
                                                                    f : Bool → α
                                                                    b : Bool
                                                                    ⊢ Eq (_root_.cond (id b) (f Bool.true) (f Bool.false)) (f b)
                                                                  -/
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
  (cond .id (const (f true)) (const (f false))).of_eq fun b => by cases b <;> rfl
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


theorem dom_bool₂ (f : Bool → Bool → α) : Primrec₂ f :=
  (cond fst ((dom_bool (f true)).comp snd) ((dom_bool (f false)).comp snd)).of_eq fun ⟨a, b⟩ => by
    /-
      α : Type u_1
      inst✝ : Primcodable α
      f : Bool → Bool → α
      x✝ : Prod Bool Bool
      a b : Bool
      ⊢ Eq (_root_.cond { fst := a, snd := b }.1 (f Bool.true { fst := a, snd := b } …
    -/
                /-
                  🎉 no goals
                -/
    cases a <;> rfl
                /-
                  🎉 no goals
                -/


protected theorem not : Primrec not :=
  dom_bool _


protected theorem and : Primrec₂ and :=
  dom_bool₂ _


protected theorem or : Primrec₂ or :=
  dom_bool₂ _


theorem _root_.PrimrecPred.not {p : α → Prop} [DecidablePred p] (hp : PrimrecPred p) :
    PrimrecPred fun a => ¬p a :=
                                          /-
                                            α : Type u_1
                                            inst✝¹ : Primcodable α
                                            p : α → Prop
                                            inst✝ : DecidablePred p
                                            hp : PrimrecPred p
                                            n : α
                                            ⊢ Eq (Decidable.decide (p n)).not (Decidable.decide ((fun a => Not (p a)) n))
                                          -/
  (Primrec.not.comp hp).of_eq fun n => by simp
                                          /-
                                            🎉 no goals
                                          -/


theorem _root_.PrimrecPred.and {p q : α → Prop} [DecidablePred p] [DecidablePred q]
    (hp : PrimrecPred p) (hq : PrimrecPred q) : PrimrecPred fun a => p a ∧ q a :=
                                             /-
                                               α : Type u_1
                                               inst✝² : Primcodable α
                                               p q : α → Prop
                                               inst✝¹ : DecidablePred p
                                               inst✝ : DecidablePred q
                                               hp : PrimrecPred p
                                               hq : PrimrecPred q
                                               n : α
                                               ⊢ Eq ((Decidable.decide (p n)).and (Decidable.decide (q n))) (Decidable.decide …
                                             -/
  (Primrec.and.comp hp hq).of_eq fun n => by simp
                                             /-
                                               🎉 no goals
                                             -/


theorem _root_.PrimrecPred.or {p q : α → Prop} [DecidablePred p] [DecidablePred q]
    (hp : PrimrecPred p) (hq : PrimrecPred q) : PrimrecPred fun a => p a ∨ q a :=
                                            /-
                                              α : Type u_1
                                              inst✝² : Primcodable α
                                              p q : α → Prop
                                              inst✝¹ : DecidablePred p
                                              inst✝ : DecidablePred q
                                              hp : PrimrecPred p
                                              hq : PrimrecPred q
                                              n : α
                                              ⊢ Eq ((Decidable.decide (p n)).or (Decidable.decide (q n))) (Decidable.decide  …
                                            -/
  (Primrec.or.comp hp hq).of_eq fun n => by simp
                                            /-
                                              🎉 no goals
                                            -/

-- Porting note: It is unclear whether we want to boolean versions
-- of these lemmas, just the prop versions, or both
-- The boolean versions are often actually easier to use
-- but did not exist in Lean 3

protected theorem beq [DecidableEq α] : Primrec₂ (@BEq.beq α _) :=
  have : PrimrecRel fun a b : ℕ => a = b :=
                                                           /-
                                                             α : Type u_1
                                                             inst✝¹ : Primcodable α
                                                             inst✝ : DecidableEq α
                                                             a : Prod Nat Nat
                                                             ⊢ Iff (And ((fun x1 x2 => LE.le x1 x2) a.1 a.2) ((fun x1 x2 => LE.le x1 x2) a. …
                                                           -/
    (PrimrecPred.and nat_le nat_le.swap).of_eq fun a => by simp [le_antisymm_iff]
                                                           /-
                                                             🎉 no goals
                                                           -/
  (this.comp₂ (Primrec.encode.comp₂ Primrec₂.left) (Primrec.encode.comp₂ Primrec₂.right)).of_eq
    fun _ _ => encode_injective.eq_iff


protected theorem eq [DecidableEq α] : PrimrecRel (@Eq α) := Primrec.beq


theorem nat_lt : PrimrecRel ((· < ·) : ℕ → ℕ → Prop) :=
                                              /-
                                                p : Prod Nat Nat
                                                ⊢ Iff (Not (LE.le p.2 p.1)) ((fun x1 x2 => LT.lt x1 x2) p.1 p.2)
                                              -/
  (nat_le.comp snd fst).not.of_eq fun p => by simp
                                              /-
                                                🎉 no goals
                                              -/


theorem option_guard {p : α → β → Prop} [∀ a b, Decidable (p a b)] (hp : PrimrecRel p) {f : α → β}
    (hf : Primrec f) : Primrec fun a => Option.guard (p a) (f a) :=
  ite (hp.comp Primrec.id hf) (option_some_iff.2 hf) (const none)


theorem option_orElse : Primrec₂ ((· <|> ·) : Option α → Option α → Option α) :=
                                                                       /-
                                                                         α : Type u_1
                                                                         inst✝ : Primcodable α
                                                                         x✝ : Prod (Option α) (Option α)
                                                                         o₁ o₂ : Option α
                                                                         ⊢ Eq (Option.casesOn { fst := o₁, snd := o₂ }.1 { fst := o₁, snd := o₂ }.2 fun …
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
  (option_casesOn fst snd (fst.comp fst).to₂).of_eq fun ⟨o₁, o₂⟩ => by cases o₁ <;> cases o₂ <;> rfl
                                                                                                 /-
                                                                                                   🎉 no goals
                                                                                                 -/


protected theorem decode₂ : Primrec (decode₂ α) :=
  option_bind .decode <|
                                        /-
                                          α : Type u_1
                                          inst✝ : Primcodable α
                                          ⊢ Primrec₂ fun p => Encodable.encode
                                        -/
                                        /-
                                          🎉 no goals
                                        -/
    option_guard (Primrec.beq.comp₂ (by exact encode_iff.mpr snd) (by exact fst.comp fst)) snd
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem list_findIdx₁ {p : α → β → Bool} (hp : Primrec₂ p) :
    ∀ l : List β, Primrec fun a => l.findIdx (p a)
| [] => const 0
| a :: l => (cond (hp.comp .id (const a)) (const 0) (succ.comp (list_findIdx₁ hp l))).of_eq fun n =>
     /-
       α : Type u_1
       β : Type u_2
       inst✝¹ : Primcodable α
       inst✝ : Primcodable β
       p : α → β → Bool
       hp : Primrec₂ p
       a : β
       l : List β
       n : α
       ⊢ Eq (_root_.cond (p (id n) a) 0 (List.findIdx (p n) l).succ) (List.findIdx (p …
     -/
  by simp [List.findIdx_cons]
     /-
       🎉 no goals
     -/


theorem list_indexOf₁ [DecidableEq α] (l : List α) : Primrec fun a => l.indexOf a :=
  list_findIdx₁ (.swap .beq) l


theorem dom_fintype [Finite α] (f : α → σ) : Primrec f :=
  let ⟨l, _, m⟩ := Finite.exists_univ_list α
  option_some_iff.1 <| by
    /-
      α : Type u_1
      σ : Type u_3
      inst✝² : Primcodable α
      inst✝¹ : Primcodable σ
      inst✝ : Finite α
      f : α → σ
      l : List α
      left✝ : l.Nodup
      m : ∀ (x : α), Membership.mem l x
      ⊢ Primrec fun a => Option.some (f a)
    -/
    haveI := decidableEqOfEncodable α
    /-
      α : Type u_1
      σ : Type u_3
      inst✝² : Primcodable α
      inst✝¹ : Primcodable σ
      inst✝ : Finite α
      f : α → σ
      l : List α
      left✝ : l.Nodup
      m : ∀ (x : α), Membership.mem l x
      this : DecidableEq α
      ⊢ Primrec fun a => Option.some (f a)
    -/
    refine ((list_get?₁ (l.map f)).comp (list_indexOf₁ l)).of_eq fun a => ?_
    /-
      α : Type u_1
      σ : Type u_3
      inst✝² : Primcodable α
      inst✝¹ : Primcodable σ
      inst✝ : Finite α
      f : α → σ
      l : List α
      left✝ : l.Nodup
      m : ∀ (x : α), Membership.mem l x
      this : DecidableEq α
      a : α
      ⊢ Eq ((List.map f l).get? (List.indexOf a l)) (Option.some (f a))
    -/
    rw [List.get?_eq_getElem?, List.getElem?_map, List.getElem?_indexOf (m a), Option.map_some']
    /-
      🎉 no goals
    -/

-- Porting note: These are new lemmas
-- I added it because it actually simplified the proofs
-- and because I couldn't understand the original proof

/-- A function is `PrimrecBounded` if its size is bounded by a primitive recursive function -/
def PrimrecBounded (f : α → β) : Prop :=
  ∃ g : α → ℕ, Primrec g ∧ ∀ x, encode (f x) ≤ g x


theorem nat_findGreatest {f : α → ℕ} {p : α → ℕ → Prop} [∀ x n, Decidable (p x n)]
    (hf : Primrec f) (hp : PrimrecRel p) : Primrec fun x => (f x).findGreatest (p x) :=
  (nat_rec' (h := fun x nih => if p x (nih.1 + 1) then nih.1 + 1 else nih.2)
    hf (const 0) (ite (hp.comp fst (snd |> fst.comp |> succ.comp))
      (snd |> fst.comp |> succ.comp) (snd.comp snd))).of_eq fun x => by
        /-
          α : Type u_1
          inst✝¹ : Primcodable α
          f : α → Nat
          p : α → Nat → Prop
          inst✝ : (x : α) → (n : Nat) → Decidable (p x n)
          hf : Primrec f
          hp : PrimrecRel p
          x : α
          ⊢ Eq (Nat.rec 0 (fun n IH => _root_.ite (p x (HAdd.hAdd { fst := n, snd := IH  …
        -/
                          /-
                            🎉 no goals
                          -/
        induction f x <;> simp [Nat.findGreatest, *]
                          /-
                            🎉 no goals
                          -/


/-- To show a function `f : α → ℕ` is primitive recursive, it is enough to show that the function
  is bounded by a primitive recursive function and that its graph is primitive recursive -/
theorem of_graph {f : α → ℕ} (h₁ : PrimrecBounded f)
    (h₂ : PrimrecRel fun a b => f a = b) : Primrec f := by
  /-
    α : Type u_1
    inst✝ : Primcodable α
    f : α → Nat
    h₁ : Primrec.PrimrecBounded f
    h₂ : PrimrecRel fun a b => Eq (f a) b
    ⊢ Primrec f
  -/
  rcases h₁ with ⟨g, pg, hg : ∀ x, f x ≤ g x⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝ : Primcodable α
    f : α → Nat
    h₂ : PrimrecRel fun a b => Eq (f a) b
    g : α → Nat
    pg : Primrec g
    hg : ∀ (x : α), LE.le (f x) (g x)
    ⊢ Primrec f
  -/
  refine (nat_findGreatest pg h₂).of_eq fun n => ?_
  /-
    case intro.intro
    α : Type u_1
    inst✝ : Primcodable α
    f : α → Nat
    h₂ : PrimrecRel fun a b => Eq (f a) b
    g : α → Nat
    pg : Primrec g
    hg : ∀ (x : α), LE.le (f x) (g x)
    n : α
    ⊢ Eq (Nat.findGreatest (fun b => Eq (f n) b) (g n)) (f n)
  -/
  exact (Nat.findGreatest_spec (P := fun b => f n = b) (hg n) rfl).symm
  /-
    🎉 no goals
  -/

-- We show that division is primitive recursive by showing that the graph is

theorem nat_div : Primrec₂ ((· / ·) : ℕ → ℕ → ℕ) := by
  /-
    ⊢ Primrec₂ fun x1 x2 => HDiv.hDiv x1 x2
  -/
  refine of_graph ⟨_, fst, fun p => Nat.div_le_self _ _⟩ ?_
  have : PrimrecRel fun (a : ℕ × ℕ) (b : ℕ) => (a.2 = 0 ∧ b = 0) ∨
      (0 < a.2 ∧ b * a.2 ≤ a.1 ∧ a.1 < (b + 1) * a.2) :=
    PrimrecPred.or
      (.and (const 0 |> Primrec.eq.comp (fst |> snd.comp)) (const 0 |> Primrec.eq.comp snd))
      (.and (nat_lt.comp (const 0) (fst |> snd.comp)) <|
          .and (nat_le.comp (nat_mul.comp snd (fst |> snd.comp)) (fst |> fst.comp))
          (nat_lt.comp (fst.comp fst) (nat_mul.comp (Primrec.succ.comp snd) (snd.comp fst))))
  /-
    this : PrimrecRel fun a b => Or (And (Eq a.2 0) (Eq b 0)) (And (LT.lt 0 a.2) ( …
    ⊢ PrimrecRel fun a b => Eq ((fun x1 x2 => HDiv.hDiv x1 x2) a.1 a.2) b
  -/
  refine this.of_eq ?_
  /-
    this : PrimrecRel fun a b => Or (And (Eq a.2 0) (Eq b 0)) (And (LT.lt 0 a.2) ( …
    ⊢ ∀ (a : Prod Nat Nat) (b : Nat), Iff (Or (And (Eq a.2 0) (Eq b 0)) (And (LT.l …
  -/
  rintro ⟨a, k⟩ q
  if H : k = 0 then simp [H, eq_comm]
  else
    have : q * k ≤ a ∧ a < (q + 1) * k ↔ q = a / k := by
      rw [le_antisymm_iff, ← (@Nat.lt_succ _ q), Nat.le_div_iff_mul_le (Nat.pos_of_ne_zero H),
          Nat.div_lt_iff_lt_mul (Nat.pos_of_ne_zero H)]
    simpa [H, zero_lt_iff, eq_comm (b := q)]


theorem nat_mod : Primrec₂ ((· % ·) : ℕ → ℕ → ℕ) :=
  (nat_sub.comp fst (nat_mul.comp snd nat_div)).to₂.of_eq fun m n => by
    /-
      m n : Nat
      ⊢ Eq (HSub.hSub { fst := m, snd := n }.1 (HMul.hMul { fst := m, snd := n }.2 ( …
    -/
    apply Nat.sub_eq_of_eq_add
    /-
      case h
      m n : Nat
      ⊢ Eq { fst := m, snd := n }.1 (HAdd.hAdd (HMod.hMod m n) (HMul.hMul { fst := m …
    -/
    simp [add_comm (m % n), Nat.div_add_mod]
    /-
      🎉 no goals
    -/


theorem nat_bodd : Primrec Nat.bodd :=
  (Primrec.beq.comp (nat_mod.comp .id (const 2)) (const 1)).of_eq fun n => by
    /-
      n : Nat
      ⊢ Eq (BEq.beq (HMod.hMod (id n) 2) 1) n.bodd
    -/
                         /-
                           🎉 no goals
                         -/
    cases H : n.bodd <;> simp [Nat.mod_two_of_bodd, H]
                         /-
                           🎉 no goals
                         -/


theorem nat_div2 : Primrec Nat.div2 :=
  (nat_div.comp .id (const 2)).of_eq fun n => n.div2_val.symm


theorem nat_double : Primrec (fun n : ℕ => 2 * n) :=
  nat_mul.comp (const _) Primrec.id


theorem nat_double_succ : Primrec (fun n : ℕ => 2 * n + 1) :=
  nat_double |> Primrec.succ.comp


private def prim : Primcodable (List β) := ⟨H⟩


private theorem list_casesOn' {f : α → List β} {g : α → σ} {h : α → β × List β → σ}
    (hf : haveI := prim H; Primrec f) (hg : Primrec g) (hh : haveI := prim H; Primrec₂ h) :
    @Primrec _ σ _ _ fun a => List.casesOn (f a) (g a) fun b l => h a (b, l) :=
  letI := prim H
  have :
    @Primrec _ (Option σ) _ _ fun a =>
      (@decode (Option (β × List β)) _ (encode (f a))).map fun o => Option.casesOn o (g a) (h a) :=
    ((@map_decode_iff _ (Option (β × List β)) _ _ _ _ _).2 <|
      to₂ <|
        option_casesOn snd (hg.comp fst) (hh.comp₂ (fst.comp₂ Primrec₂.left) Primrec₂.right)).comp
      .id (encode_iff.2 hf)
                                              /-
                                                α : Type u_1
                                                β : Type u_2
                                                σ : Type u_3
                                                inst✝² : Primcodable α
                                                inst✝¹ : Primcodable β
                                                inst✝ : Primcodable σ
                                                H : Nat.Primrec fun n => Encodable.encode (Encodable.decode n)
                                                f : α → List β
                                                g : α → σ
                                                h : α → Prod β (List β) → σ
                                                hf : Primrec f
                                                hg : Primrec g
                                                hh : Primrec₂ h
                                                this✝ : Primcodable (List β) := prim H
                                                this : Primrec fun a => Option.map (fun o => Option.casesOn o (g a) (h a)) (En …
                                                a : α
                                                ⊢ Eq (Option.map (fun o => Option.casesOn o (g a) (h a)) (Encodable.decode (En …
                                              -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
  option_some_iff.1 <| this.of_eq fun a => by cases' f a with b l <;> simp [encodek]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


private theorem list_foldl' {f : α → List β} {g : α → σ} {h : α → σ × β → σ}
    (hf : haveI := prim H; Primrec f) (hg : Primrec g) (hh : haveI := prim H; Primrec₂ h) :
    Primrec fun a => (f a).foldl (fun s b => h a (s, b)) (g a) := by
  /-
    α : Type u_1
    β : Type u_2
    σ : Type u_3
    inst✝² : Primcodable α
    inst✝¹ : Primcodable β
    inst✝ : Primcodable σ
    H : Nat.Primrec fun n => Encodable.encode (Encodable.decode n)
    f : α → List β
    g : α → σ
    h : α → Prod σ β → σ
    hf : Primrec f
    hg : Primrec g
    hh : Primrec₂ h
    ⊢ Primrec fun a => List.foldl (fun s b => h a { fst := s, snd := b }) (g a) (f …
  -/
  letI := prim H
  /-
    α : Type u_1
    β : Type u_2
    σ : Type u_3
    inst✝² : Primcodable α
    inst✝¹ : Primcodable β
    inst✝ : Primcodable σ
    H : Nat.Primrec fun n => Encodable.encode (Encodable.decode n)
    f : α → List β
    g : α → σ
    h : α → Prod σ β → σ
    hf : Primrec f
    hg : Primrec g
    hh : Primrec₂ h
    this : Primcodable (List β) := prim H
    ⊢ Primrec fun a => List.foldl (fun s b => h a { fst := s, snd := b }) (g a) (f …
  -/
  let G (a : α) (IH : σ × List β) : σ × List β := List.casesOn IH.2 IH fun b l => (h a (IH.1, b), l)
  have hG : Primrec₂ G := list_casesOn' H (snd.comp snd) snd <|
    to₂ <|
    pair (hh.comp (fst.comp fst) <| pair ((fst.comp snd).comp fst) (fst.comp snd))
      (snd.comp snd)
  /-
    α : Type u_1
    β : Type u_2
    σ : Type u_3
    inst✝² : Primcodable α
    inst✝¹ : Primcodable β
    inst✝ : Primcodable σ
    H : Nat.Primrec fun n => Encodable.encode (Encodable.decode n)
    f : α → List β
    g : α → σ
    h : α → Prod σ β → σ
    hf : Primrec f
    hg : Primrec g
    hh : Primrec₂ h
    this : Primcodable (List β) := prim H
    G : α → Prod σ (List β) → Prod σ (List β) := fun a IH => List.casesOn IH.2 IH  …
    hG : Primrec₂ G
    ⊢ Primrec fun a => List.foldl (fun s b => h a { fst := s, snd := b }) (g a) (f …
  -/
  let F := fun (a : α) (n : ℕ) => (G a)^[n] (g a, f a)
  have hF : Primrec fun a => (F a (encode (f a))).1 :=
    (fst.comp <|
      nat_iterate (encode_iff.2 hf) (pair hg hf) <|
      hG)
  suffices ∀ a n, F a n = (((f a).take n).foldl (fun s b => h a (s, b)) (g a), (f a).drop n) by
    refine hF.of_eq fun a => ?_
    rw [this, List.take_of_length_le (length_le_encode _)]
  /-
    α : Type u_1
    β : Type u_2
    σ : Type u_3
    inst✝² : Primcodable α
    inst✝¹ : Primcodable β
    inst✝ : Primcodable σ
    H : Nat.Primrec fun n => Encodable.encode (Encodable.decode n)
    f : α → List β
    g : α → σ
    h : α → Prod σ β → σ
    hf : Primrec f
    hg : Primrec g
    hh : Primrec₂ h
    this : Primcodable (List β) := prim H
    G : α → Prod σ (List β) → Prod σ (List β) := fun a IH => List.casesOn IH.2 IH  …
    hG : Primrec₂ G
    F : α → Nat → Prod σ (List β) := fun a n => Nat.iterate (G a) n { fst := g a,  …
    hF : Primrec fun a => (F a (Encodable.encode (f a))).1
    ⊢ ∀ (a : α) (n : Nat), Eq (F a n) { fst := List.foldl (fun s b => h a { fst := …
  -/
  introv
  /-
    α : Type u_1
    β : Type u_2
    σ : Type u_3
    inst✝² : Primcodable α
    inst✝¹ : Primcodable β
    inst✝ : Primcodable σ
    H : Nat.Primrec fun n => Encodable.encode (Encodable.decode n)
    f : α → List β
    g : α → σ
    h : α → Prod σ β → σ
    hf : Primrec f
    hg : Primrec g
    hh : Primrec₂ h
    this : Primcodable (List β) := prim H
    G : α → Prod σ (List β) → Prod σ (List β) := fun a IH => List.casesOn IH.2 IH  …
    hG : Primrec₂ G
    F : α → Nat → Prod σ (List β) := fun a n => Nat.iterate (G a) n { fst := g a,  …
    hF : Primrec fun a => (F a (Encodable.encode (f a))).1
    a : α
    n : Nat
    ⊢ Eq (F a n) { fst := List.foldl (fun s b => h a { fst := s, snd := b }) (g a) …
  -/
  dsimp only [F]
  /-
    α : Type u_1
    β : Type u_2
    σ : Type u_3
    inst✝² : Primcodable α
    inst✝¹ : Primcodable β
    inst✝ : Primcodable σ
    H : Nat.Primrec fun n => Encodable.encode (Encodable.decode n)
    f : α → List β
    g : α → σ
    h : α → Prod σ β → σ
    hf : Primrec f
    hg : Primrec g
    hh : Primrec₂ h
    this : Primcodable (List β) := prim H
    G : α → Prod σ (List β) → Prod σ (List β) := fun a IH => List.casesOn IH.2 IH  …
    hG : Primrec₂ G
    F : α → Nat → Prod σ (List β) := fun a n => Nat.iterate (G a) n { fst := g a,  …
    hF : Primrec fun a => (F a (Encodable.encode (f a))).1
    a : α
    n : Nat
    ⊢ Eq (Nat.iterate (G a) n { fst := g a, snd := f a }) { fst := List.foldl (fun …
  -/
  generalize f a = l
  /-
    α : Type u_1
    β : Type u_2
    σ : Type u_3
    inst✝² : Primcodable α
    inst✝¹ : Primcodable β
    inst✝ : Primcodable σ
    H : Nat.Primrec fun n => Encodable.encode (Encodable.decode n)
    f : α → List β
    g : α → σ
    h : α → Prod σ β → σ
    hf : Primrec f
    hg : Primrec g
    hh : Primrec₂ h
    this : Primcodable (List β) := prim H
    G : α → Prod σ (List β) → Prod σ (List β) := fun a IH => List.casesOn IH.2 IH  …
    hG : Primrec₂ G
    F : α → Nat → Prod σ (List β) := fun a n => Nat.iterate (G a) n { fst := g a,  …
    hF : Primrec fun a => (F a (Encodable.encode (f a))).1
    a : α
    n : Nat
    l : List β
    ⊢ Eq (Nat.iterate (G a) n { fst := g a, snd := l }) { fst := List.foldl (fun s …
  -/
  generalize g a = x
  induction n generalizing l x with
  | zero => rfl
  | succ n IH =>
    simp only [iterate_succ, comp_apply]
    cases' l with b l <;> simp [G, IH]


private theorem list_cons' : (haveI := prim H; Primrec₂ (@List.cons β)) :=
  letI := prim H
  encode_iff.1 (succ.comp <| Primrec₂.natPair.comp (encode_iff.2 fst) (encode_iff.2 snd))


private theorem list_reverse' :
    haveI := prim H
    Primrec (@List.reverse β) :=
  letI := prim H
  (list_foldl' H .id (const []) <| to₂ <| ((list_cons' H).comp snd fst).comp snd).of_eq
    (suffices ∀ l r, List.foldl (fun (s : List β) (b : β) => b :: s) r l = List.reverseAux l r from
      fun l => this l []
                /-
                  β : Type u_2
                  inst✝ : Primcodable β
                  H : Nat.Primrec fun n => Encodable.encode (Encodable.decode n)
                  this : Primcodable (List β) := prim H
                  l : List β
                  ⊢ ∀ (r : List β), Eq (List.foldl (fun s b => List.cons b s) r l) (l.reverseAux …
                -/
                                /-
                                  🎉 no goals
                                -/
    fun l => by induction l <;> simp [*, List.reverseAux])
                                /-
                                  🎉 no goals
                                -/


instance sum : Primcodable (α ⊕ β) :=
  ⟨Primrec.nat_iff.1 <|
      (encode_iff.2
            (cond nat_bodd
              (((@Primrec.decode β _).comp nat_div2).option_map <|
                to₂ <| nat_double_succ.comp (Primrec.encode.comp snd))
              (((@Primrec.decode α _).comp nat_div2).option_map <|
                to₂ <| nat_double.comp (Primrec.encode.comp snd)))).of_eq
        fun n =>
        show _ = encode (decodeSum n) by
          /-
            α : Type u_1
            β : Type u_2
            inst✝¹ : Primcodable α
            inst✝ : Primcodable β
            n : Nat
            ⊢ Eq (Encodable.encode (cond n.bodd (Option.map (fun b => HAdd.hAdd (HMul.hMul …
          -/
          simp only [decodeSum, Nat.boddDiv2_eq]
          /-
            α : Type u_1
            β : Type u_2
            inst✝¹ : Primcodable α
            inst✝ : Primcodable β
            n : Nat
            ⊢ Eq (Encodable.encode (cond n.bodd (Option.map (fun b => HAdd.hAdd (HMul.hMul …
          -/
          cases Nat.bodd n <;> simp [decodeSum]
            /-
              case false
              α : Type u_1
              β : Type u_2
              inst✝¹ : Primcodable α
              inst✝ : Primcodable β
              n : Nat
              ⊢ Eq (Encodable.encode (Option.map (fun b => HMul.hMul 2 (Encodable.encode b)) …
            -/
                                         /-
                                           🎉 no goals
                                         -/
          · cases @decode α _ n.div2 <;> rfl
                                         /-
                                           🎉 no goals
                                         -/
            /-
              case true
              α : Type u_1
              β : Type u_2
              inst✝¹ : Primcodable α
              inst✝ : Primcodable β
              n : Nat
              ⊢ Eq (Encodable.encode (Option.map (fun b => HAdd.hAdd (HMul.hMul 2 (Encodable …
            -/
                                         /-
                                           🎉 no goals
                                         -/
          · cases @decode β _ n.div2 <;> rfl⟩
                                         /-
                                           🎉 no goals
                                         -/


instance list : Primcodable (List α) :=
  ⟨letI H := @Primcodable.prim (List ℕ) _
    have : Primrec₂ fun (a : α) (o : Option (List ℕ)) => o.map (List.cons (encode a)) :=
      option_map snd <| (list_cons' H).comp ((@Primrec.encode α _).comp (fst.comp fst)) snd
    have :
      Primrec fun n =>
        (ofNat (List ℕ) n).reverse.foldl
          (fun o m => (@decode α _ m).bind fun a => o.map (List.cons (encode a))) (some []) :=
      list_foldl' H ((list_reverse' H).comp (.ofNat (List ℕ))) (const (some []))
        (Primrec.comp₂ (bind_decode_iff.2 <| .swap this) Primrec₂.right)
    nat_iff.1 <|
      (encode_iff.2 this).of_eq fun n => by
        /-
          α : Type u_1
          β : Type u_2
          inst✝¹ : Primcodable α
          inst✝ : Primcodable β
          H : Nat.Primrec fun n => Encodable.encode (Encodable.decode n) := Primcodable. …
          this✝ : Primrec₂ fun a o => Option.map (List.cons (Encodable.encode a)) o
          this : Primrec fun n => List.foldl (fun o m => (Encodable.decode m).bind fun a …
          n : Nat
          ⊢ Eq (Encodable.encode (List.foldl (fun o m => (Encodable.decode m).bind fun a …
        -/
        rw [List.foldl_reverse]
        /-
          α : Type u_1
          β : Type u_2
          inst✝¹ : Primcodable α
          inst✝ : Primcodable β
          H : Nat.Primrec fun n => Encodable.encode (Encodable.decode n) := Primcodable. …
          this✝ : Primrec₂ fun a o => Option.map (List.cons (Encodable.encode a)) o
          this : Primrec fun n => List.foldl (fun o m => (Encodable.decode m).bind fun a …
          n : Nat
          ⊢ Eq (Encodable.encode (List.foldr (fun x y => (Encodable.decode x).bind fun a …
        -/
        apply Nat.case_strong_induction_on n; · simp
                                                /-
                                                  🎉 no goals
                                                -/
        /-
          case hi
          α : Type u_1
          β : Type u_2
          inst✝¹ : Primcodable α
          inst✝ : Primcodable β
          H : Nat.Primrec fun n => Encodable.encode (Encodable.decode n) := Primcodable. …
          this✝ : Primrec₂ fun a o => Option.map (List.cons (Encodable.encode a)) o
          this : Primrec fun n => List.foldl (fun o m => (Encodable.decode m).bind fun a …
          n : Nat
          ⊢ ∀ (n : Nat), (∀ (m : Nat), LE.le m n → Eq (Encodable.encode (List.foldr (fun …
        -/
        intro n IH; simp
        /-
          case hi
          α : Type u_1
          β : Type u_2
          inst✝¹ : Primcodable α
          inst✝ : Primcodable β
          H : Nat.Primrec fun n => Encodable.encode (Encodable.decode n) := Primcodable. …
          this✝ : Primrec₂ fun a o => Option.map (List.cons (Encodable.encode a)) o
          this : Primrec fun n => List.foldl (fun o m => (Encodable.decode m).bind fun a …
          n✝ n : Nat
          IH : ∀ (m : Nat), LE.le m n → Eq (Encodable.encode (List.foldr (fun x y => (En …
          ⊢ Eq (Encodable.encode ((Encodable.decode (Nat.unpair n).1).bind fun a => Opti …
        -/
        cases' @decode α _ n.unpair.1 with a; · rfl
                                                /-
                                                  🎉 no goals
                                                -/
        /-
          case hi.some
          α : Type u_1
          β : Type u_2
          inst✝¹ : Primcodable α
          inst✝ : Primcodable β
          H : Nat.Primrec fun n => Encodable.encode (Encodable.decode n) := Primcodable. …
          this✝ : Primrec₂ fun a o => Option.map (List.cons (Encodable.encode a)) o
          this : Primrec fun n => List.foldl (fun o m => (Encodable.decode m).bind fun a …
          n✝ n : Nat
          IH : ∀ (m : Nat), LE.le m n → Eq (Encodable.encode (List.foldr (fun x y => (En …
          a : α
          ⊢ Eq (Encodable.encode ((Option.some a).bind fun a => Option.map (List.cons (E …
        -/
        simp only [decode_eq_ofNat, Option.some.injEq, Option.some_bind, Option.map_some']
        suffices ∀ (o : Option (List ℕ)) (p), encode o = encode p →
            encode (Option.map (List.cons (encode a)) o) = encode (Option.map (List.cons a) p) from
          this _ _ (IH _ (Nat.unpair_right_le n))
        /-
          case hi.some
          α : Type u_1
          β : Type u_2
          inst✝¹ : Primcodable α
          inst✝ : Primcodable β
          H : Nat.Primrec fun n => Encodable.encode (Encodable.decode n) := Primcodable. …
          this✝ : Primrec₂ fun a o => Option.map (List.cons (Encodable.encode a)) o
          this : Primrec fun n => List.foldl (fun o m => (Encodable.decode m).bind fun a …
          n✝ n : Nat
          IH : ∀ (m : Nat), LE.le m n → Eq (Encodable.encode (List.foldr (fun x y => (En …
          a : α
          ⊢ ∀ (o : Option (List Nat)) (p : Option (List α)), Eq (Encodable.encode o) (En …
        -/
        intro o p IH
        /-
          case hi.some
          α : Type u_1
          β : Type u_2
          inst✝¹ : Primcodable α
          inst✝ : Primcodable β
          H : Nat.Primrec fun n => Encodable.encode (Encodable.decode n) := Primcodable. …
          this✝ : Primrec₂ fun a o => Option.map (List.cons (Encodable.encode a)) o
          this : Primrec fun n => List.foldl (fun o m => (Encodable.decode m).bind fun a …
          n✝ n : Nat
          IH✝ : ∀ (m : Nat), LE.le m n → Eq (Encodable.encode (List.foldr (fun x y => (E …
          a : α
          o : Option (List Nat)
          p : Option (List α)
          IH : Eq (Encodable.encode o) (Encodable.encode p)
          ⊢ Eq (Encodable.encode (Option.map (List.cons (Encodable.encode a)) o)) (Encod …
        -/
        cases o <;> cases p
          /-
            case hi.some.none.none
            α : Type u_1
            β : Type u_2
            inst✝¹ : Primcodable α
            inst✝ : Primcodable β
            H : Nat.Primrec fun n => Encodable.encode (Encodable.decode n) := Primcodable. …
            this✝ : Primrec₂ fun a o => Option.map (List.cons (Encodable.encode a)) o
            this : Primrec fun n => List.foldl (fun o m => (Encodable.decode m).bind fun a …
            n✝ n : Nat
            IH✝ : ∀ (m : Nat), LE.le m n → Eq (Encodable.encode (List.foldr (fun x y => (E …
            a : α
            IH : Eq (Encodable.encode Option.none) (Encodable.encode Option.none)
            ⊢ Eq (Encodable.encode (Option.map (List.cons (Encodable.encode a)) Option.non …
          -/
        · rfl
          /-
            🎉 no goals
          -/
          /-
            case hi.some.none.some
            α : Type u_1
            β : Type u_2
            inst✝¹ : Primcodable α
            inst✝ : Primcodable β
            H : Nat.Primrec fun n => Encodable.encode (Encodable.decode n) := Primcodable. …
            this✝ : Primrec₂ fun a o => Option.map (List.cons (Encodable.encode a)) o
            this : Primrec fun n => List.foldl (fun o m => (Encodable.decode m).bind fun a …
            n✝ n : Nat
            IH✝ : ∀ (m : Nat), LE.le m n → Eq (Encodable.encode (List.foldr (fun x y => (E …
            a : α
            val✝ : List α
            IH : Eq (Encodable.encode Option.none) (Encodable.encode (Option.some val✝))
            ⊢ Eq (Encodable.encode (Option.map (List.cons (Encodable.encode a)) Option.non …
          -/
        · injection IH
          /-
            🎉 no goals
          -/
          /-
            case hi.some.some.none
            α : Type u_1
            β : Type u_2
            inst✝¹ : Primcodable α
            inst✝ : Primcodable β
            H : Nat.Primrec fun n => Encodable.encode (Encodable.decode n) := Primcodable. …
            this✝ : Primrec₂ fun a o => Option.map (List.cons (Encodable.encode a)) o
            this : Primrec fun n => List.foldl (fun o m => (Encodable.decode m).bind fun a …
            n✝ n : Nat
            IH✝ : ∀ (m : Nat), LE.le m n → Eq (Encodable.encode (List.foldr (fun x y => (E …
            a : α
            val✝ : List Nat
            IH : Eq (Encodable.encode (Option.some val✝)) (Encodable.encode Option.none)
            ⊢ Eq (Encodable.encode (Option.map (List.cons (Encodable.encode a)) (Option.so …
          -/
        · injection IH
          /-
            🎉 no goals
          -/
          /-
            case hi.some.some.some
            α : Type u_1
            β : Type u_2
            inst✝¹ : Primcodable α
            inst✝ : Primcodable β
            H : Nat.Primrec fun n => Encodable.encode (Encodable.decode n) := Primcodable. …
            this✝ : Primrec₂ fun a o => Option.map (List.cons (Encodable.encode a)) o
            this : Primrec fun n => List.foldl (fun o m => (Encodable.decode m).bind fun a …
            n✝ n : Nat
            IH✝ : ∀ (m : Nat), LE.le m n → Eq (Encodable.encode (List.foldr (fun x y => (E …
            a : α
            val✝¹ : List Nat
            val✝ : List α
            IH : Eq (Encodable.encode (Option.some val✝¹)) (Encodable.encode (Option.some  …
            ⊢ Eq (Encodable.encode (Option.map (List.cons (Encodable.encode a)) (Option.so …
          -/
        · exact congr_arg (fun k => (Nat.pair (encode a) k).succ.succ) (Nat.succ.inj IH)⟩
          /-
            🎉 no goals
          -/

theorem sum_inl : Primrec (@Sum.inl α β) :=
  encode_iff.1 <| nat_double.comp Primrec.encode


theorem sum_inr : Primrec (@Sum.inr α β) :=
  encode_iff.1 <| nat_double_succ.comp Primrec.encode


theorem sum_casesOn {f : α → β ⊕ γ} {g : α → β → σ} {h : α → γ → σ} (hf : Primrec f)
    (hg : Primrec₂ g) (hh : Primrec₂ h) : @Primrec _ σ _ _ fun a => Sum.casesOn (f a) (g a) (h a) :=
  option_some_iff.1 <|
    (cond (nat_bodd.comp <| encode_iff.2 hf)
          (option_map (Primrec.decode.comp <| nat_div2.comp <| encode_iff.2 hf) hh)
          (option_map (Primrec.decode.comp <| nat_div2.comp <| encode_iff.2 hf) hg)).of_eq
                  /-
                    α : Type u_1
                    β : Type u_2
                    γ : Type u_3
                    σ : Type u_4
                    inst✝³ : Primcodable α
                    inst✝² : Primcodable β
                    inst✝¹ : Primcodable γ
                    inst✝ : Primcodable σ
                    f : α → Sum β γ
                    g : α → β → σ
                    h : α → γ → σ
                    hf : Primrec f
                    hg : Primrec₂ g
                    hh : Primrec₂ h
                    a : α
                    ⊢ Eq (_root_.cond (Encodable.encode (f a)).bodd (Option.map (h a) (Encodable.d …
                  -/
                                          /-
                                            🎉 no goals
                                          -/
      fun a => by cases' f a with b c <;> simp [Nat.div2_val, encodek]
                                          /-
                                            🎉 no goals
                                          -/


theorem list_cons : Primrec₂ (@List.cons α) :=
  list_cons' Primcodable.prim


theorem list_casesOn {f : α → List β} {g : α → σ} {h : α → β × List β → σ} :
    Primrec f →
      Primrec g →
        Primrec₂ h → @Primrec _ σ _ _ fun a => List.casesOn (f a) (g a) fun b l => h a (b, l) :=
  list_casesOn' Primcodable.prim


theorem list_foldl {f : α → List β} {g : α → σ} {h : α → σ × β → σ} :
    Primrec f →
      Primrec g → Primrec₂ h → Primrec fun a => (f a).foldl (fun s b => h a (s, b)) (g a) :=
  list_foldl' Primcodable.prim


theorem list_reverse : Primrec (@List.reverse α) :=
  list_reverse' Primcodable.prim


theorem list_foldr {f : α → List β} {g : α → σ} {h : α → β × σ → σ} (hf : Primrec f)
    (hg : Primrec g) (hh : Primrec₂ h) :
    Primrec fun a => (f a).foldr (fun b s => h a (b, s)) (g a) :=
  (list_foldl (list_reverse.comp hf) hg <| to₂ <| hh.comp fst <| (pair snd fst).comp snd).of_eq
                /-
                  α : Type u_1
                  β : Type u_2
                  σ : Type u_4
                  inst✝² : Primcodable α
                  inst✝¹ : Primcodable β
                  inst✝ : Primcodable σ
                  f : α → List β
                  g : α → σ
                  h : α → Prod β σ → σ
                  hf : Primrec f
                  hg : Primrec g
                  hh : Primrec₂ h
                  a : α
                  ⊢ Eq (List.foldl (fun s b => h { fst := a, snd := { fst := s, snd := b } }.1 { …
                -/
    fun a => by simp [List.foldl_reverse]
                /-
                  🎉 no goals
                -/


theorem list_head? : Primrec (@List.head? α) :=
  (list_casesOn .id (const none) (option_some_iff.2 <| fst.comp snd).to₂).of_eq fun l => by
    /-
      α : Type u_1
      inst✝ : Primcodable α
      l : List α
      ⊢ Eq (List.casesOn (id l) Option.none fun b l_1 => Option.some { fst := l, snd …
    -/
                /-
                  🎉 no goals
                -/
    cases l <;> rfl
                /-
                  🎉 no goals
                -/


theorem list_headI [Inhabited α] : Primrec (@List.headI α _) :=
  (option_iget.comp list_head?).of_eq fun l => l.head!_eq_head?.symm


theorem list_tail : Primrec (@List.tail α) :=
                                                                     /-
                                                                       α : Type u_1
                                                                       inst✝ : Primcodable α
                                                                       l : List α
                                                                       ⊢ Eq (List.casesOn (id l) List.nil fun b l_1 => { fst := l, snd := { fst := b, …
                                                                     -/
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
  (list_casesOn .id (const []) (snd.comp snd).to₂).of_eq fun l => by cases l <;> rfl
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


theorem list_rec {f : α → List β} {g : α → σ} {h : α → β × List β × σ → σ} (hf : Primrec f)
    (hg : Primrec g) (hh : Primrec₂ h) :
    @Primrec _ σ _ _ fun a => List.recOn (f a) (g a) fun b l IH => h a (b, l, IH) :=
  let F (a : α) := (f a).foldr (fun (b : β) (s : List β × σ) => (b :: s.1, h a (b, s))) ([], g a)
  have : Primrec F :=
    list_foldr hf (pair (const []) hg) <|
      to₂ <| pair ((list_cons.comp fst (fst.comp snd)).comp snd) hh
  (snd.comp this).of_eq fun a => by
    /-
      α : Type u_1
      β : Type u_2
      σ : Type u_4
      inst✝² : Primcodable α
      inst✝¹ : Primcodable β
      inst✝ : Primcodable σ
      f : α → List β
      g : α → σ
      h : α → Prod β (Prod (List β) σ) → σ
      hf : Primrec f
      hg : Primrec g
      hh : Primrec₂ h
      F : α → Prod (List β) σ := fun a => List.foldr (fun b s => { fst := List.cons  …
      this : Primrec F
      a : α
      ⊢ Eq (F a).2 (List.recOn (f a) (g a) fun b l IH => h a { fst := b, snd := { fs …
    -/
    suffices F a = (f a, List.recOn (f a) (g a) fun b l IH => h a (b, l, IH)) by rw [this]
    /-
      α : Type u_1
      β : Type u_2
      σ : Type u_4
      inst✝² : Primcodable α
      inst✝¹ : Primcodable β
      inst✝ : Primcodable σ
      f : α → List β
      g : α → σ
      h : α → Prod β (Prod (List β) σ) → σ
      hf : Primrec f
      hg : Primrec g
      hh : Primrec₂ h
      F : α → Prod (List β) σ := fun a => List.foldr (fun b s => { fst := List.cons  …
      this : Primrec F
      a : α
      ⊢ Eq (F a) { fst := f a, snd := List.recOn (f a) (g a) fun b l IH => h a { fst …
    -/
    dsimp [F]
    /-
      α : Type u_1
      β : Type u_2
      σ : Type u_4
      inst✝² : Primcodable α
      inst✝¹ : Primcodable β
      inst✝ : Primcodable σ
      f : α → List β
      g : α → σ
      h : α → Prod β (Prod (List β) σ) → σ
      hf : Primrec f
      hg : Primrec g
      hh : Primrec₂ h
      F : α → Prod (List β) σ := fun a => List.foldr (fun b s => { fst := List.cons  …
      this : Primrec F
      a : α
      ⊢ Eq (List.foldr (fun b s => { fst := List.cons b s.1, snd := h a { fst := b,  …
    -/
                                   /-
                                     🎉 no goals
                                   -/
    induction' f a with b l IH <;> simp [*]
                                   /-
                                     🎉 no goals
                                   -/


theorem list_get? : Primrec₂ (@List.get? α) :=
  let F (l : List α) (n : ℕ) :=
    l.foldl
      (fun (s : ℕ ⊕ α) (a : α) =>
        Sum.casesOn s (@Nat.casesOn (fun _ => ℕ ⊕ α) · (Sum.inr a) Sum.inl) Sum.inr)
      (Sum.inl n)
  have hF : Primrec₂ F :=
    (list_foldl fst (sum_inl.comp snd)
      ((sum_casesOn fst (nat_casesOn snd (sum_inr.comp <| snd.comp fst) (sum_inl.comp snd).to₂).to₂
              (sum_inr.comp snd).to₂).comp
          snd).to₂).to₂
  have :
    @Primrec _ (Option α) _ _ fun p : List α × ℕ => Sum.casesOn (F p.1 p.2) (fun _ => none) some :=
    sum_casesOn hF (const none).to₂ (option_some.comp snd).to₂
  this.to₂.of_eq fun l n => by
    /-
      α : Type u_1
      inst✝ : Primcodable α
      F : List α → Nat → Sum Nat α := fun l n => List.foldl (fun s a => Sum.casesOn  …
      hF : Primrec₂ F
      this : Primrec fun p => Sum.casesOn (F p.1 p.2) (fun x => Option.none) Option. …
      l : List α
      n : Nat
      ⊢ Eq (Sum.casesOn (F { fst := l, snd := n }.1 { fst := l, snd := n }.2) (fun x …
    -/
    dsimp; symm
    /-
      α : Type u_1
      inst✝ : Primcodable α
      F : List α → Nat → Sum Nat α := fun l n => List.foldl (fun s a => Sum.casesOn  …
      hF : Primrec₂ F
      this : Primrec fun p => Sum.casesOn (F p.1 p.2) (fun x => Option.none) Option. …
      l : List α
      n : Nat
      ⊢ Eq (l.get? n) (Sum.rec (fun val => Option.none) (fun val => Option.some val) …
    -/
    induction' l with a l IH generalizing n; · rfl
                                               /-
                                                 🎉 no goals
                                               -/
    /-
      case cons
      α : Type u_1
      inst✝ : Primcodable α
      F : List α → Nat → Sum Nat α := fun l n => List.foldl (fun s a => Sum.casesOn  …
      hF : Primrec₂ F
      this : Primrec fun p => Sum.casesOn (F p.1 p.2) (fun x => Option.none) Option. …
      a : α
      l : List α
      IH : ∀ (n : Nat), Eq (l.get? n) (Sum.rec (fun val => Option.none) (fun val =>  …
      n : Nat
      ⊢ Eq ((List.cons a l).get? n) (Sum.rec (fun val => Option.none) (fun val => Op …
    -/
    cases' n with n
      /-
        case cons.zero
        α : Type u_1
        inst✝ : Primcodable α
        F : List α → Nat → Sum Nat α := fun l n => List.foldl (fun s a => Sum.casesOn  …
        hF : Primrec₂ F
        this : Primrec fun p => Sum.casesOn (F p.1 p.2) (fun x => Option.none) Option. …
        a : α
        l : List α
        IH : ∀ (n : Nat), Eq (l.get? n) (Sum.rec (fun val => Option.none) (fun val =>  …
        ⊢ Eq ((List.cons a l).get? 0) (Sum.rec (fun val => Option.none) (fun val => Op …
      -/
    · dsimp [F]
      /-
        case cons.zero
        α : Type u_1
        inst✝ : Primcodable α
        F : List α → Nat → Sum Nat α := fun l n => List.foldl (fun s a => Sum.casesOn  …
        hF : Primrec₂ F
        this : Primrec fun p => Sum.casesOn (F p.1 p.2) (fun x => Option.none) Option. …
        a : α
        l : List α
        IH : ∀ (n : Nat), Eq (l.get? n) (Sum.rec (fun val => Option.none) (fun val =>  …
        ⊢ Eq (Option.some a) (Sum.rec (fun val => Option.none) (fun val => Option.some …
      -/
      clear IH
      /-
        case cons.zero
        α : Type u_1
        inst✝ : Primcodable α
        F : List α → Nat → Sum Nat α := fun l n => List.foldl (fun s a => Sum.casesOn  …
        hF : Primrec₂ F
        this : Primrec fun p => Sum.casesOn (F p.1 p.2) (fun x => Option.none) Option. …
        a : α
        l : List α
        ⊢ Eq (Option.some a) (Sum.rec (fun val => Option.none) (fun val => Option.some …
      -/
                                   /-
                                     🎉 no goals
                                   -/
      induction' l with _ l IH <;> simp [*]
                                   /-
                                     🎉 no goals
                                   -/
      /-
        case cons.succ
        α : Type u_1
        inst✝ : Primcodable α
        F : List α → Nat → Sum Nat α := fun l n => List.foldl (fun s a => Sum.casesOn  …
        hF : Primrec₂ F
        this : Primrec fun p => Sum.casesOn (F p.1 p.2) (fun x => Option.none) Option. …
        a : α
        l : List α
        IH : ∀ (n : Nat), Eq (l.get? n) (Sum.rec (fun val => Option.none) (fun val =>  …
        n : Nat
        ⊢ Eq ((List.cons a l).get? (HAdd.hAdd n 1)) (Sum.rec (fun val => Option.none)  …
      -/
    · apply IH
      /-
        🎉 no goals
      -/


theorem list_getElem? : Primrec₂ (fun (l : List α) (n : ℕ) => l[n]?) := by
  /-
    α : Type u_1
    inst✝ : Primcodable α
    ⊢ Primrec₂ fun l n => GetElem?.getElem? l n
  -/
  convert list_get?
  /-
    case h.e'_7.h.h.h.e
    α : Type u_1
    inst✝ : Primcodable α
    x✝¹ : List α
    x✝ : Nat
    ⊢ Eq GetElem?.getElem? List.get?
  -/
  ext
  /-
    case h.e'_7.h.h.h.e.h.h.a
    α : Type u_1
    inst✝ : Primcodable α
    x✝³ : List α
    x✝² : Nat
    x✝¹ : List α
    x✝ : Nat
    a✝ : α
    ⊢ Iff (Membership.mem (GetElem?.getElem? x✝¹ x✝) a✝) (Membership.mem (x✝¹.get? …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem list_getD (d : α) : Primrec₂ fun l n => List.getD l n d := by
  /-
    α : Type u_1
    inst✝ : Primcodable α
    d : α
    ⊢ Primrec₂ fun l n => l.getD n d
  -/
  simp only [List.getD_eq_getElem?_getD]
  /-
    α : Type u_1
    inst✝ : Primcodable α
    d : α
    ⊢ Primrec₂ fun l n => (GetElem?.getElem? l n).getD d
  -/
  exact option_getD.comp₂ list_getElem? (const _)
  /-
    🎉 no goals
  -/


theorem list_getI [Inhabited α] : Primrec₂ (@List.getI α _) :=
  list_getD _


theorem list_append : Primrec₂ ((· ++ ·) : List α → List α → List α) :=
  (list_foldr fst snd <| to₂ <| comp (@list_cons α _) snd).to₂.of_eq fun l₁ l₂ => by
    /-
      α : Type u_1
      inst✝ : Primcodable α
      l₁ l₂ : List α
      ⊢ Eq (List.foldr (fun b s => List.cons { fst := { fst := l₁, snd := l₂ }, snd  …
    -/
                     /-
                       🎉 no goals
                     -/
    induction l₁ <;> simp [*]
                     /-
                       🎉 no goals
                     -/


theorem list_concat : Primrec₂ fun l (a : α) => l ++ [a] :=
  list_append.comp fst (list_cons.comp snd (const []))


theorem list_map {f : α → List β} {g : α → β → σ} (hf : Primrec f) (hg : Primrec₂ g) :
    Primrec fun a => (f a).map (g a) :=
  (list_foldr hf (const []) <|
        to₂ <| list_cons.comp (hg.comp fst (fst.comp snd)) (snd.comp snd)).of_eq
                /-
                  α : Type u_1
                  β : Type u_2
                  σ : Type u_4
                  inst✝² : Primcodable α
                  inst✝¹ : Primcodable β
                  inst✝ : Primcodable σ
                  f : α → List β
                  g : α → β → σ
                  hf : Primrec f
                  hg : Primrec₂ g
                  a : α
                  ⊢ Eq (List.foldr (fun b s => List.cons (g { fst := a, snd := { fst := b, snd : …
                -/
                                  /-
                                    🎉 no goals
                                  -/
    fun a => by induction f a <;> simp [*]
                                  /-
                                    🎉 no goals
                                  -/


theorem list_range : Primrec List.range :=
  (nat_rec' .id (const []) ((list_concat.comp snd fst).comp snd).to₂).of_eq fun n => by
    /-
      n : Nat
      ⊢ Eq (Nat.rec List.nil (fun n_1 IH => HAppend.hAppend { fst := n, snd := { fst …
    -/
                          /-
                            🎉 no goals
                          -/
    simp; induction n <;> simp [*, List.range_succ]
                          /-
                            🎉 no goals
                          -/


theorem list_flatten : Primrec (@List.flatten α) :=
  (list_foldr .id (const []) <| to₂ <| comp (@list_append α _) snd).of_eq fun l => by
    /-
      α : Type u_1
      inst✝ : Primcodable α
      l : List (List α)
      ⊢ Eq (List.foldr (fun b s => (fun x1 x2 => HAppend.hAppend x1 x2) { fst := l,  …
    -/
                           /-
                             🎉 no goals
                           -/
    dsimp; induction l <;> simp [*]
                           /-
                             🎉 no goals
                           -/


@[deprecated (since := "2024-10-15")] alias list_join := list_flatten


theorem list_flatMap {f : α → List β} {g : α → β → List σ} (hf : Primrec f) (hg : Primrec₂ g) :
    Primrec (fun a => (f a).flatMap (g a)) := list_flatten.comp (list_map hf hg)


@[deprecated (since := "2024-10-16")] alias list_bind := list_flatMap


theorem optionToList : Primrec (Option.toList : Option α → List α) :=
  (option_casesOn Primrec.id (const [])
    ((list_cons.comp Primrec.id (const [])).comp₂ Primrec₂.right)).of_eq
               /-
                 α : Type u_1
                 inst✝ : Primcodable α
                 o : Option α
                 ⊢ Eq (Option.casesOn (id o) List.nil fun b => List.cons (id b) List.nil) o.toL …
               -/
                            /-
                              🎉 no goals
                            -/
  (fun o => by rcases o <;> simp)
                            /-
                              🎉 no goals
                            -/


theorem listFilterMap {f : α → List β} {g : α → β → Option σ}
    (hf : Primrec f) (hg : Primrec₂ g) : Primrec fun a => (f a).filterMap (g a) :=
  (list_flatMap hf (comp₂ optionToList hg)).of_eq
    fun _ ↦ Eq.symm <| List.filterMap_eq_flatMap_toList _ _


theorem list_length : Primrec (@List.length α) :=
  (list_foldr (@Primrec.id (List α) _) (const 0) <| to₂ <| (succ.comp <| snd.comp snd).to₂).of_eq
                /-
                  α : Type u_1
                  inst✝ : Primcodable α
                  l : List α
                  ⊢ Eq (List.foldr (fun b s => (fun a b => { fst := a, snd := b }.2.2.succ) { fs …
                -/
                                       /-
                                         🎉 no goals
                                       -/
    fun l => by dsimp; induction l <;> simp [*]
                                       /-
                                         🎉 no goals
                                       -/


theorem list_findIdx {f : α → List β} {p : α → β → Bool}
    (hf : Primrec f) (hp : Primrec₂ p) : Primrec fun a => (f a).findIdx (p a) :=
  (list_foldr hf (const 0) <|
        to₂ <| cond (hp.comp fst <| fst.comp snd) (const 0) (succ.comp <| snd.comp snd)).of_eq
                /-
                  α : Type u_1
                  β : Type u_2
                  inst✝¹ : Primcodable α
                  inst✝ : Primcodable β
                  f : α → List β
                  p : α → β → Bool
                  hf : Primrec f
                  hp : Primrec₂ p
                  a : α
                  ⊢ Eq (List.foldr (fun b s => _root_.cond (p { fst := a, snd := { fst := b, snd …
                -/
                                         /-
                                           🎉 no goals
                                         -/
    fun a => by dsimp; induction f a <;> simp [List.findIdx_cons, *]
                                         /-
                                           🎉 no goals
                                         -/


theorem list_indexOf [DecidableEq α] : Primrec₂ (@List.indexOf α _) :=
  to₂ <| list_findIdx snd <| Primrec.beq.comp₂ snd.to₂ (fst.comp fst).to₂


theorem nat_strong_rec (f : α → ℕ → σ) {g : α → List σ → Option σ} (hg : Primrec₂ g)
    (H : ∀ a n, g a ((List.range n).map (f a)) = some (f a n)) : Primrec₂ f :=
  suffices Primrec₂ fun a n => (List.range n).map (f a) from
    Primrec₂.option_some_iff.1 <|
      (list_get?.comp (this.comp fst (succ.comp snd)) snd).to₂.of_eq fun a n => by
        /-
          α : Type u_1
          σ : Type u_4
          inst✝¹ : Primcodable α
          inst✝ : Primcodable σ
          f : α → Nat → σ
          g : α → List σ → Option σ
          hg : Primrec₂ g
          H : ∀ (a : α) (n : Nat), Eq (g a (List.map (f a) (List.range n))) (Option.some …
          this : Primrec₂ fun a n => List.map (f a) (List.range n)
          a : α
          n : Nat
          ⊢ Eq ((List.map (f { fst := a, snd := n }.1) (List.range { fst := a, snd := n  …
        -/
        simp [List.getElem?_range (Nat.lt_succ_self n)]
        /-
          🎉 no goals
        -/
  Primrec₂.option_some_iff.1 <|
    (nat_rec (const (some []))
          (to₂ <|
            option_bind (snd.comp snd) <|
              to₂ <|
                option_map (hg.comp (fst.comp fst) snd)
                  (to₂ <| list_concat.comp (snd.comp fst) snd))).of_eq
      fun a n => by
      induction n with
      | zero => rfl
      | succ n IH => simp [IH, H, List.range_succ]


theorem listLookup [DecidableEq α] : Primrec₂ (List.lookup : α → List (α × β) → Option β) :=
  (to₂ <| list_rec snd (const none) <|
    to₂ <|
      cond (Primrec.beq.comp (fst.comp fst) (fst.comp <| fst.comp snd))
        (option_some.comp <| snd.comp <| fst.comp snd)
        (snd.comp <| snd.comp snd)).of_eq
  fun a ps => by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : Primcodable α
    inst✝¹ : Primcodable β
    inst✝ : DecidableEq α
    a : α
    ps : List (Prod α β)
    ⊢ Eq (List.recOn { fst := a, snd := ps }.2 Option.none fun b l IH => _root_.co …
  -/
                                 /-
                                   🎉 no goals
                                 -/
  induction' ps with p ps ih <;> simp [List.lookup, *]
  /-
    case cons
    α : Type u_1
    β : Type u_2
    inst✝² : Primcodable α
    inst✝¹ : Primcodable β
    inst✝ : DecidableEq α
    a : α
    p : Prod α β
    ps : List (Prod α β)
    ih : Eq (List.recOn { fst := a, snd := ps }.2 Option.none fun b l IH => _root_ …
    ⊢ Eq (_root_.cond (BEq.beq a p.1) (Option.some p.2) (List.lookup a ps)) (List. …
  -/
                          /-
                            🎉 no goals
                          -/
  cases ha : a == p.1 <;> simp [ha]
                          /-
                            🎉 no goals
                          -/


theorem nat_omega_rec' (f : β → σ) {m : β → ℕ} {l : β → List β} {g : β → List σ → Option σ}
    (hm : Primrec m) (hl : Primrec l) (hg : Primrec₂ g)
    (Ord : ∀ b, ∀ b' ∈ l b, m b' < m b)
    (H : ∀ b, g b ((l b).map f) = some (f b)) : Primrec f := by
  /-
    β : Type u_2
    σ : Type u_4
    inst✝¹ : Primcodable β
    inst✝ : Primcodable σ
    f : β → σ
    m : β → Nat
    l : β → List β
    g : β → List σ → Option σ
    hm : Primrec m
    hl : Primrec l
    hg : Primrec₂ g
    Ord : ∀ (b b' : β), Membership.mem (l b) b' → LT.lt (m b') (m b)
    H : ∀ (b : β), Eq (g b (List.map f (l b))) (Option.some (f b))
    ⊢ Primrec f
  -/
  haveI : DecidableEq β := Encodable.decidableEqOfEncodable β
  /-
    β : Type u_2
    σ : Type u_4
    inst✝¹ : Primcodable β
    inst✝ : Primcodable σ
    f : β → σ
    m : β → Nat
    l : β → List β
    g : β → List σ → Option σ
    hm : Primrec m
    hl : Primrec l
    hg : Primrec₂ g
    Ord : ∀ (b b' : β), Membership.mem (l b) b' → LT.lt (m b') (m b)
    H : ∀ (b : β), Eq (g b (List.map f (l b))) (Option.some (f b))
    this : DecidableEq β
    ⊢ Primrec f
  -/
  let mapGraph (M : List (β × σ)) (bs : List β) : List σ := bs.flatMap (Option.toList <| M.lookup ·)
  /-
    β : Type u_2
    σ : Type u_4
    inst✝¹ : Primcodable β
    inst✝ : Primcodable σ
    f : β → σ
    m : β → Nat
    l : β → List β
    g : β → List σ → Option σ
    hm : Primrec m
    hl : Primrec l
    hg : Primrec₂ g
    Ord : ∀ (b b' : β), Membership.mem (l b) b' → LT.lt (m b') (m b)
    H : ∀ (b : β), Eq (g b (List.map f (l b))) (Option.some (f b))
    this : DecidableEq β
    mapGraph : List (Prod β σ) → List β → List σ := fun M bs => bs.flatMap fun x = …
    ⊢ Primrec f
  -/
  let bindList (b : β) : ℕ → List β := fun n ↦ n.rec [b] fun _ bs ↦ bs.flatMap l
  let graph (b : β) : ℕ → List (β × σ) := fun i ↦ i.rec [] fun i ih ↦
    (bindList b (m b - i)).filterMap fun b' ↦ (g b' <| mapGraph ih (l b')).map (b', ·)
  have mapGraph_primrec : Primrec₂ mapGraph :=
    to₂ <| list_flatMap snd <| optionToList.comp₂ <| listLookup.comp₂ .right (fst.comp₂ .left)
  have bindList_primrec : Primrec₂ (bindList) :=
    nat_rec' snd
      (list_cons.comp fst (const []))
      (to₂ <| list_flatMap (snd.comp snd) (hl.comp₂ .right))
  have graph_primrec : Primrec₂ (graph) :=
    to₂ <| nat_rec' snd (const []) <|
      to₂ <| listFilterMap
        (bindList_primrec.comp
          (fst.comp fst)
          (nat_sub.comp (hm.comp <| fst.comp fst) (fst.comp snd))) <|
            to₂ <| option_map
              (hg.comp snd (mapGraph_primrec.comp (snd.comp <| snd.comp fst) (hl.comp snd)))
              (Primrec₂.pair.comp₂ (snd.comp₂ .left) .right)
  have : Primrec (fun b => ((graph b (m b + 1)).get? 0).map Prod.snd) :=
    option_map (list_get?.comp (graph_primrec.comp Primrec.id (succ.comp hm)) (const 0))
      (snd.comp₂ Primrec₂.right)
  exact option_some_iff.mp <| this.of_eq <| fun b ↦ by
    have graph_eq_map_bindList (i : ℕ) (hi : i ≤ m b + 1) :
        graph b i = (bindList b (m b + 1 - i)).map fun x ↦ (x, f x) := by
      have bindList_eq_nil : bindList b (m b + 1) = [] :=
        have bindList_m_lt (k : ℕ) : ∀ b' ∈ bindList b k, m b' < m b + 1 - k := by
          induction' k with k ih <;> simp [bindList]
          intro a₂ a₁ ha₁ ha₂
          have : k ≤ m b :=
            Nat.lt_succ.mp (by simpa using Nat.add_lt_of_lt_sub <| Nat.zero_lt_of_lt (ih a₁ ha₁))
          have : m a₁ ≤ m b - k :=
            Nat.lt_succ.mp (by rw [← Nat.succ_sub this]; simpa using ih a₁ ha₁)
          exact lt_of_lt_of_le (Ord a₁ a₂ ha₂) this
        List.eq_nil_iff_forall_not_mem.mpr
          (by intro b' ha'; by_contra; simpa using bindList_m_lt (m b + 1) b' ha')
      have mapGraph_graph {bs bs' : List β} (has : bs' ⊆ bs) :
          mapGraph (bs.map <| fun x => (x, f x)) bs' = bs'.map f := by
        induction' bs' with b bs' ih <;> simp [mapGraph]
        · have : b ∈ bs ∧ bs' ⊆ bs := by simpa using has
          rcases this with ⟨ha, has'⟩
          simpa [List.lookup_graph f ha] using ih has'
      have graph_succ : ∀ i, graph b (i + 1) =
        (bindList b (m b - i)).filterMap fun b' =>
          (g b' <| mapGraph (graph b i) (l b')).map (b', ·) := fun _ => rfl
      have bindList_succ : ∀ i, bindList b (i + 1) = (bindList b i).flatMap l := fun _ => rfl
      induction' i with i ih
      · symm; simpa [graph] using bindList_eq_nil
      · simp only [graph_succ, ih (Nat.le_of_lt hi), Nat.succ_sub (Nat.lt_succ.mp hi),
          Nat.succ_eq_add_one, bindList_succ, Nat.reduceSubDiff]
        apply List.filterMap_eq_map_iff_forall_eq_some.mpr
        intro b' ha'; simp; rw [mapGraph_graph]
        · exact H b'
        · exact (List.infix_flatMap_of_mem ha' l).subset
    simp [graph_eq_map_bindList (m b + 1) (Nat.le_refl _), bindList]


theorem nat_omega_rec (f : α → β → σ) {m : α → β → ℕ}
    {l : α → β → List β} {g : α → β × List σ → Option σ}
    (hm : Primrec₂ m) (hl : Primrec₂ l) (hg : Primrec₂ g)
    (Ord : ∀ a b, ∀ b' ∈ l a b, m a b' < m a b)
    (H : ∀ a b, g a (b, (l a b).map (f a)) = some (f a b)) : Primrec₂ f :=
  Primrec₂.uncurry.mp <|
    nat_omega_rec' (Function.uncurry f)
      (Primrec₂.uncurry.mpr hm)
      (list_map (hl.comp fst snd) (Primrec₂.pair.comp₂ (fst.comp₂ .left) .right))
      (hg.comp₂ (fst.comp₂ .left) (Primrec₂.pair.comp₂ (snd.comp₂ .left) .right))
          /-
            α : Type u_1
            β : Type u_2
            σ : Type u_4
            inst✝² : Primcodable α
            inst✝¹ : Primcodable β
            inst✝ : Primcodable σ
            f : α → β → σ
            m : α → β → Nat
            l : α → β → List β
            g : α → Prod β (List σ) → Option σ
            hm : Primrec₂ m
            hl : Primrec₂ l
            hg : Primrec₂ g
            Ord : ∀ (a : α) (b b' : β), Membership.mem (l a b) b' → LT.lt (m a b') (m a b)
            H : ∀ (a : α) (b : β), Eq (g a { fst := b, snd := List.map (f a) (l a b) }) (O …
            ⊢ ∀ (b b' : Prod α β), Membership.mem (List.map (fun b_1 => { fst := b.1, snd  …
          -/
          /-
            🎉 no goals
          -/
      (by simpa using Ord) (by simpa [Function.comp] using H)
                               /-
                                 🎉 no goals
                               -/


/-- A subtype of a primitive recursive predicate is `Primcodable`. -/
def subtype {p : α → Prop} [DecidablePred p] (hp : PrimrecPred p) : Primcodable (Subtype p) :=
  ⟨have : Primrec fun n => (@decode α _ n).bind fun a => Option.guard p a :=
    option_bind .decode (option_guard (hp.comp snd).to₂ snd)
  nat_iff.1 <| (encode_iff.2 this).of_eq fun n =>
    show _ = encode ((@decode α _ n).bind fun _ => _) by
      /-
        α : Type u_1
        inst✝¹ : Primcodable α
        p : α → Prop
        inst✝ : DecidablePred p
        hp : PrimrecPred p
        this : Primrec fun n => (Encodable.decode n).bind fun a => Option.guard p a
        n : Nat
        ⊢ Eq (Encodable.encode ((Encodable.decode n).bind fun a => Option.guard p a))  …
      -/
      cases' @decode α _ n with a; · rfl
                                     /-
                                       🎉 no goals
                                     -/
      /-
        case some
        α : Type u_1
        inst✝¹ : Primcodable α
        p : α → Prop
        inst✝ : DecidablePred p
        hp : PrimrecPred p
        this : Primrec fun n => (Encodable.decode n).bind fun a => Option.guard p a
        n : Nat
        a : α
        ⊢ Eq (Encodable.encode ((Option.some a).bind fun a => Option.guard p a)) (Enco …
      -/
      dsimp [Option.guard]
      /-
        case some
        α : Type u_1
        inst✝¹ : Primcodable α
        p : α → Prop
        inst✝ : DecidablePred p
        hp : PrimrecPred p
        this : Primrec fun n => (Encodable.decode n).bind fun a => Option.guard p a
        n : Nat
        a : α
        ⊢ Eq (Encodable.encode (ite (p a) (Option.some a) Option.none)) (Encodable.enc …
      -/
                           /-
                             🎉 no goals
                           -/
      by_cases h : p a <;> simp [h]; rfl⟩
                                     /-
                                       🎉 no goals
                                     -/


instance fin {n} : Primcodable (Fin n) :=
  @ofEquiv _ _ (subtype <| nat_lt.comp .id (const n)) Fin.equivSubtype


instance vector {n} : Primcodable (List.Vector α n) :=
  subtype ((@Primrec.eq ℕ _ _).comp list_length (const _))


instance finArrow {n} : Primcodable (Fin n → α) :=
  ofEquiv _ (Equiv.vectorEquivFin _ _).symm

-- Porting note: Equiv.arrayEquivFin is not ported yet
-- instance array {n} : Primcodable (Array' n α) :=
--   ofEquiv _ (Equiv.arrayEquivFin _ _)


theorem mem_range_encode : PrimrecPred (fun n => n ∈ Set.range (encode : α → ℕ)) :=
  have : PrimrecPred fun n => Encodable.decode₂ α n ≠ none :=
    .not
      (Primrec.eq.comp
        (.option_bind .decode
          (.ite (Primrec.eq.comp (Primrec.encode.comp .snd) .fst)
            (Primrec.option_some.comp .snd) (.const _)))
        (.const _))
  this.of_eq fun _ => decode₂_ne_none_iff


instance ulower : Primcodable (ULower α) :=
  Primcodable.subtype mem_range_encode


theorem subtype_val {p : α → Prop} [DecidablePred p] {hp : PrimrecPred p} :
    haveI := Primcodable.subtype hp
    Primrec (@Subtype.val α p) := by
  /-
    α : Type u_1
    inst✝¹ : Primcodable α
    p : α → Prop
    inst✝ : DecidablePred p
    hp : PrimrecPred p
    ⊢ Primrec Subtype.val
  -/
  letI := Primcodable.subtype hp
  /-
    α : Type u_1
    inst✝¹ : Primcodable α
    p : α → Prop
    inst✝ : DecidablePred p
    hp : PrimrecPred p
    this : Primcodable (Subtype p) := Primcodable.subtype hp
    ⊢ Primrec Subtype.val
  -/
  refine (@Primcodable.prim (Subtype p)).of_eq fun n => ?_
  /-
    α : Type u_1
    inst✝¹ : Primcodable α
    p : α → Prop
    inst✝ : DecidablePred p
    hp : PrimrecPred p
    this : Primcodable (Subtype p) := Primcodable.subtype hp
    n : Nat
    ⊢ Eq (Encodable.encode (Encodable.decode n)) (Encodable.encode (Option.map Sub …
  -/
                                                       /-
                                                         🎉 no goals
                                                       -/
  rcases @decode (Subtype p) _ n with (_ | ⟨a, h⟩) <;> rfl
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem subtype_val_iff {p : β → Prop} [DecidablePred p] {hp : PrimrecPred p} {f : α → Subtype p} :
    haveI := Primcodable.subtype hp
    (Primrec fun a => (f a).1) ↔ Primrec f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : Primcodable α
    inst✝¹ : Primcodable β
    p : β → Prop
    inst✝ : DecidablePred p
    hp : PrimrecPred p
    f : α → Subtype p
    ⊢ Iff (Primrec fun a => ↑(f a)) (Primrec f)
  -/
  letI := Primcodable.subtype hp
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : Primcodable α
    inst✝¹ : Primcodable β
    p : β → Prop
    inst✝ : DecidablePred p
    hp : PrimrecPred p
    f : α → Subtype p
    this : Primcodable (Subtype p) := Primcodable.subtype hp
    ⊢ Iff (Primrec fun a => ↑(f a)) (Primrec f)
  -/
  refine ⟨fun h => ?_, fun hf => subtype_val.comp hf⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : Primcodable α
    inst✝¹ : Primcodable β
    p : β → Prop
    inst✝ : DecidablePred p
    hp : PrimrecPred p
    f : α → Subtype p
    this : Primcodable (Subtype p) := Primcodable.subtype hp
    h : Primrec fun a => ↑(f a)
    ⊢ Primrec f
  -/
  refine Nat.Primrec.of_eq h fun n => ?_
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : Primcodable α
    inst✝¹ : Primcodable β
    p : β → Prop
    inst✝ : DecidablePred p
    hp : PrimrecPred p
    f : α → Subtype p
    this : Primcodable (Subtype p) := Primcodable.subtype hp
    h : Primrec fun a => ↑(f a)
    n : Nat
    ⊢ Eq (Encodable.encode (Option.map (fun a => ↑(f a)) (Encodable.decode n))) (E …
  -/
  cases' @decode α _ n with a; · rfl
                                 /-
                                   🎉 no goals
                                 -/
  /-
    case some
    α : Type u_1
    β : Type u_2
    inst✝² : Primcodable α
    inst✝¹ : Primcodable β
    p : β → Prop
    inst✝ : DecidablePred p
    hp : PrimrecPred p
    f : α → Subtype p
    this : Primcodable (Subtype p) := Primcodable.subtype hp
    h : Primrec fun a => ↑(f a)
    n : Nat
    a : α
    ⊢ Eq (Encodable.encode (Option.map (fun a => ↑(f a)) (Option.some a))) (Encoda …
  -/
  simp; rfl
        /-
          🎉 no goals
        -/


theorem subtype_mk {p : β → Prop} [DecidablePred p] {hp : PrimrecPred p} {f : α → β}
    {h : ∀ a, p (f a)} (hf : Primrec f) :
    haveI := Primcodable.subtype hp
    Primrec fun a => @Subtype.mk β p (f a) (h a) :=
  subtype_val_iff.1 hf


theorem option_get {f : α → Option β} {h : ∀ a, (f a).isSome} :
    Primrec f → Primrec fun a => (f a).get (h a) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Primcodable α
    inst✝ : Primcodable β
    f : α → Option β
    h : ∀ (a : α), Eq (f a).isSome Bool.true
    ⊢ Primrec f → Primrec fun a => (f a).get ⋯
  -/
  intro hf
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Primcodable α
    inst✝ : Primcodable β
    f : α → Option β
    h : ∀ (a : α), Eq (f a).isSome Bool.true
    hf : Primrec f
    ⊢ Primrec fun a => (f a).get ⋯
  -/
  refine (Nat.Primrec.pred.comp hf).of_eq fun n => ?_
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Primcodable α
    inst✝ : Primcodable β
    f : α → Option β
    h : ∀ (a : α), Eq (f a).isSome Bool.true
    hf : Primrec f
    n : Nat
    ⊢ Eq (Encodable.encode (Option.map f (Encodable.decode n))).pred (Encodable.en …
  -/
  generalize hx : @decode α _ n = x
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Primcodable α
    inst✝ : Primcodable β
    f : α → Option β
    h : ∀ (a : α), Eq (f a).isSome Bool.true
    hf : Primrec f
    n : Nat
    x : Option α
    hx : Eq (Encodable.decode n) x
    ⊢ Eq (Encodable.encode (Option.map f x)).pred (Encodable.encode (Option.map (f …
  -/
              /-
                🎉 no goals
              -/
  cases x <;> simp
              /-
                🎉 no goals
              -/


theorem ulower_down : Primrec (ULower.down : α → ULower α) :=
  letI : ∀ a, Decidable (a ∈ Set.range (encode : α → ℕ)) := decidableRangeEncode _
  subtype_mk .encode


theorem ulower_up : Primrec (ULower.up : ULower α → α) :=
  letI : ∀ a, Decidable (a ∈ Set.range (encode : α → ℕ)) := decidableRangeEncode _
  option_get (Primrec.decode₂.comp subtype_val)


theorem fin_val_iff {n} {f : α → Fin n} : (Primrec fun a => (f a).1) ↔ Primrec f := by
  /-
    α : Type u_1
    inst✝ : Primcodable α
    n : Nat
    f : α → Fin n
    ⊢ Iff (Primrec fun a => ↑(f a)) (Primrec f)
  -/
  letI : Primcodable { a // id a < n } := Primcodable.subtype (nat_lt.comp .id (const _))
  /-
    α : Type u_1
    inst✝ : Primcodable α
    n : Nat
    f : α → Fin n
    this : Primcodable (Subtype fun a => LT.lt (id a) n) := Primcodable.subtype ⋯
    ⊢ Iff (Primrec fun a => ↑(f a)) (Primrec f)
  -/
  exact (Iff.trans (by rfl) subtype_val_iff).trans (of_equiv_iff _)
  /-
    🎉 no goals
  -/


theorem fin_val {n} : Primrec (fun (i : Fin n) => (i : ℕ)) :=
  fin_val_iff.2 .id


theorem fin_succ {n} : Primrec (@Fin.succ n) :=
                      /-
                        n : Nat
                        ⊢ Primrec fun a => ↑a.succ
                      -/
  fin_val_iff.1 <| by simp [succ.comp fin_val]
                      /-
                        🎉 no goals
                      -/


theorem vector_toList {n} : Primrec (@List.Vector.toList α n) :=
  subtype_val


theorem vector_toList_iff {n} {f : α → List.Vector β n} :
    (Primrec fun a => (f a).toList) ↔ Primrec f :=
  subtype_val_iff


theorem vector_cons {n} : Primrec₂ (@List.Vector.cons α n) :=
                            /-
                              α : Type u_1
                              inst✝ : Primcodable α
                              n : Nat
                              ⊢ Primrec fun a => (List.Vector.cons a.1 a.2).toList
                            -/
  vector_toList_iff.1 <| by simpa using list_cons.comp fst (vector_toList_iff.2 snd)
                            /-
                              🎉 no goals
                            -/


theorem vector_length {n} : Primrec (@List.Vector.length α n) :=
  const _


theorem vector_head {n} : Primrec (@List.Vector.head α n) :=
  option_some_iff.1 <| (list_head?.comp vector_toList).of_eq fun ⟨_ :: _, _⟩ => rfl


theorem vector_tail {n} : Primrec (@List.Vector.tail α n) :=
                                                                               /-
                                                                                 α : Type u_1
                                                                                 inst✝ : Primcodable α
                                                                                 n : Nat
                                                                                 x✝ : List.Vector α n
                                                                                 l : List α
                                                                                 h : Eq l.length n
                                                                                 ⊢ Eq (List.Vector.toList ⟨l, h⟩).tail (List.Vector.tail ⟨l, h⟩).toList
                                                                               -/
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/
  vector_toList_iff.1 <| (list_tail.comp vector_toList).of_eq fun ⟨l, h⟩ => by cases l <;> rfl
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/


theorem vector_get {n} : Primrec₂ (@List.Vector.get α n) :=
  option_some_iff.1 <|
    (list_get?.comp (vector_toList.comp fst) (fin_val.comp snd)).of_eq fun a => by
      /-
        α : Type u_1
        inst✝ : Primcodable α
        n : Nat
        a : Prod (List.Vector α n) (Fin n)
        ⊢ Eq (a.1.toList.get? ↑a.2) (Option.some (a.1.get a.2))
      -/
      rw [Vector.get_eq_get_toList, ← List.get?_eq_get]
      /-
        α : Type u_1
        inst✝ : Primcodable α
        n : Nat
        a : Prod (List.Vector α n) (Fin n)
        ⊢ Eq (a.1.toList.get? ↑a.2) (a.1.toList.get? ↑(Fin.cast ⋯ a.2))
      -/
      rfl
      /-
        🎉 no goals
      -/


theorem list_ofFn :
    ∀ {n} {f : Fin n → α → σ}, (∀ i, Primrec (f i)) → Primrec fun a => List.ofFn fun i => f i a
                  /-
                    α : Type u_1
                    σ : Type u_3
                    inst✝¹ : Primcodable α
                    inst✝ : Primcodable σ
                    x✝¹ : Fin 0 → α → σ
                    x✝ : ∀ (i : Fin 0), Primrec (x✝¹ i)
                    ⊢ Primrec fun a => List.ofFn fun i => x✝¹ i a
                  -/
  | 0, _, _ => by simp only [List.ofFn_zero]; exact const []
                                              /-
                                                🎉 no goals
                                              -/
  | n + 1, f, hf => by
    /-
      α : Type u_1
      σ : Type u_3
      inst✝¹ : Primcodable α
      inst✝ : Primcodable σ
      n : Nat
      f : Fin (HAdd.hAdd n 1) → α → σ
      hf : ∀ (i : Fin (HAdd.hAdd n 1)), Primrec (f i)
      ⊢ Primrec fun a => List.ofFn fun i => f i a
    -/
    simpa [List.ofFn_succ] using list_cons.comp (hf 0) (list_ofFn fun i => hf i.succ)
    /-
      🎉 no goals
    -/


theorem vector_ofFn {n} {f : Fin n → α → σ} (hf : ∀ i, Primrec (f i)) :
    Primrec fun a => List.Vector.ofFn fun i => f i a :=
                            /-
                              α : Type u_1
                              σ : Type u_3
                              inst✝¹ : Primcodable α
                              inst✝ : Primcodable σ
                              n : Nat
                              f : Fin n → α → σ
                              hf : ∀ (i : Fin n), Primrec (f i)
                              ⊢ Primrec fun a => (List.Vector.ofFn fun i => f i a).toList
                            -/
  vector_toList_iff.1 <| by simp [list_ofFn hf]
                            /-
                              🎉 no goals
                            -/


theorem vector_get' {n} : Primrec (@List.Vector.get α n) :=
  of_equiv_symm


theorem vector_ofFn' {n} : Primrec (@List.Vector.ofFn α n) :=
  of_equiv


theorem fin_app {n} : Primrec₂ (@id (Fin n → σ)) :=
                                                                       /-
                                                                         σ : Type u_3
                                                                         inst✝ : Primcodable σ
                                                                         n : Nat
                                                                         x✝ : Prod (Fin n → σ) (Fin n)
                                                                         v : Fin n → σ
                                                                         i : Fin n
                                                                         ⊢ Eq ((List.Vector.ofFn { fst := v, snd := i }.1).get { fst := v, snd := i }.2 …
                                                                       -/
  (vector_get.comp (vector_ofFn'.comp fst) snd).of_eq fun ⟨v, i⟩ => by simp
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem fin_curry₁ {n} {f : Fin n → α → σ} : Primrec₂ f ↔ ∀ i, Primrec (f i) :=
  ⟨fun h i => h.comp (const i) .id, fun h =>
                                                                       /-
                                                                         α : Type u_1
                                                                         σ : Type u_3
                                                                         inst✝¹ : Primcodable α
                                                                         inst✝ : Primcodable σ
                                                                         n : Nat
                                                                         f : Fin n → α → σ
                                                                         h : ∀ (i : Fin n), Primrec (f i)
                                                                         a : Prod (Fin n) α
                                                                         ⊢ Eq ((List.Vector.ofFn fun i => f i a.2).get a.1) (f a.1 a.2)
                                                                       -/
    (vector_get.comp ((vector_ofFn h).comp snd) fst).of_eq fun a => by simp⟩
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem fin_curry {n} {f : α → Fin n → σ} : Primrec f ↔ Primrec₂ f :=
  ⟨fun h => fin_app.comp (h.comp fst) snd, fun h =>
    (vector_get'.comp
          (vector_ofFn fun i => show Primrec fun a => f a i from h.comp .id (const i))).of_eq
                  /-
                    α : Type u_1
                    σ : Type u_3
                    inst✝¹ : Primcodable α
                    inst✝ : Primcodable σ
                    n : Nat
                    f : α → Fin n → σ
                    h : Primrec₂ f
                    a : α
                    ⊢ Eq (List.Vector.ofFn fun i => f a i).get (f a)
                  -/
      fun a => by funext i; simp⟩
                            /-
                              🎉 no goals
                            -/


/-- An alternative inductive definition of `Primrec` which
  does not use the pairing function on ℕ, and so has to
  work with n-ary functions on ℕ instead of unary functions.
  We prove that this is equivalent to the regular notion
  in `to_prim` and `of_prim`. -/
inductive Primrec' : ∀ {n}, (List.Vector ℕ n → ℕ) → Prop
  | zero : @Primrec' 0 fun _ => 0
  | succ : @Primrec' 1 fun v => succ v.head
  | get {n} (i : Fin n) : Primrec' fun v => v.get i
  | comp {m n f} (g : Fin n → List.Vector ℕ m → ℕ) :
      Primrec' f → (∀ i, Primrec' (g i)) → Primrec' fun a => f (List.Vector.ofFn fun i => g i a)
  | prec {n f g} :
      @Primrec' n f →
        @Primrec' (n + 2) g →
          Primrec' fun v : List.Vector ℕ (n + 1) =>
            v.head.rec (f v.tail) fun y IH => g (y ::ᵥ IH ::ᵥ v.tail)


theorem to_prim {n f} (pf : @Nat.Primrec' n f) : Primrec f := by
  induction pf with
  | zero => exact .const 0
  | succ => exact _root_.Primrec.succ.comp .vector_head
  | get i => exact Primrec.vector_get.comp .id (.const i)
  | comp _ _ _ hf hg => exact hf.comp (.vector_ofFn fun i => hg i)
  | @prec n f g _ _ hf hg =>
    exact
      .nat_rec' .vector_head (hf.comp Primrec.vector_tail)
        (hg.comp <|
          Primrec.vector_cons.comp (Primrec.fst.comp .snd) <|
          Primrec.vector_cons.comp (Primrec.snd.comp .snd) <|
            (@Primrec.vector_tail _ _ (n + 1)).comp .fst).to₂


theorem of_eq {n} {f g : List.Vector ℕ n → ℕ} (hf : Primrec' f) (H : ∀ i, f i = g i) :
    Primrec' g :=
  (funext H : f = g) ▸ hf


theorem const {n} : ∀ m, @Primrec' n fun _ => m
  | 0 => zero.comp Fin.elim0 fun i => i.elim0
  | m + 1 => succ.comp _ fun _ => const m


theorem head {n : ℕ} : @Primrec' n.succ head :=
                            /-
                              n : Nat
                              v : List.Vector Nat n.succ
                              ⊢ Eq (v.get 0) v.head
                            -/
  (get 0).of_eq fun v => by simp [get_zero]
                            /-
                              🎉 no goals
                            -/


theorem tail {n f} (hf : @Primrec' n f) : @Primrec' n.succ fun v => f v.tail :=
  (hf.comp _ fun i => @get _ i.succ).of_eq fun v => by
    /-
      n : Nat
      f : List.Vector Nat n → Nat
      hf : Nat.Primrec' f
      v : List.Vector Nat (HAdd.hAdd n 1)
      ⊢ Eq (f (List.Vector.ofFn fun i => v.get i.succ)) (f v.tail)
    -/
    rw [← ofFn_get v.tail]; congr; funext i; simp
                                             /-
                                               🎉 no goals
                                             -/


/-- A function from vectors to vectors is primitive recursive when all of its projections are. -/
def Vec {n m} (f : List.Vector ℕ n → List.Vector ℕ m) : Prop :=
  ∀ i, Primrec' fun v => (f v).get i


protected theorem nil {n} : @Vec n 0 fun _ => nil := fun i => i.elim0


protected theorem cons {n m f g} (hf : @Primrec' n f) (hg : @Vec n m g) :
                                                       /-
                                                         n m : Nat
                                                         f : List.Vector Nat n → Nat
                                                         g : List.Vector Nat n → List.Vector Nat m
                                                         hf : Nat.Primrec' f
                                                         hg : Nat.Primrec'.Vec g
                                                         i : Fin m.succ
                                                         ⊢ Nat.Primrec' fun v => ((fun v => List.Vector.cons (f v) (g v)) v).get 0
                                                       -/
                                                       /-
                                                         🎉 no goals
                                                       -/
    Vec fun v => f v ::ᵥ g v := fun i => Fin.cases (by simp [*]) (fun i => by simp [hg i]) i
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


theorem idv {n} : @Vec n n id :=
  get


theorem comp' {n m f g} (hf : @Primrec' m f) (hg : @Vec n m g) : Primrec' fun v => f (g v) :=
                                   /-
                                     n m : Nat
                                     f : List.Vector Nat m → Nat
                                     g : List.Vector Nat n → List.Vector Nat m
                                     hf : Nat.Primrec' f
                                     hg : Nat.Primrec'.Vec g
                                     v : List.Vector Nat n
                                     ⊢ Eq (f (List.Vector.ofFn fun i => (g v).get i)) (f (g v))
                                   -/
  (hf.comp _ hg).of_eq fun v => by simp
                                   /-
                                     🎉 no goals
                                   -/


theorem comp₁ (f : ℕ → ℕ) (hf : @Primrec' 1 fun v => f v.head) {n g} (hg : @Primrec' n g) :
    Primrec' fun v => f (g v) :=
  hf.comp _ fun _ => hg


theorem comp₂ (f : ℕ → ℕ → ℕ) (hf : @Primrec' 2 fun v => f v.head v.tail.head) {n g h}
    (hg : @Primrec' n g) (hh : @Primrec' n h) : Primrec' fun v => f (g v) (h v) := by
  /-
    f : Nat → Nat → Nat
    hf : Nat.Primrec' fun v => f v.head v.tail.head
    n : Nat
    g h : List.Vector Nat n → Nat
    hg : Nat.Primrec' g
    hh : Nat.Primrec' h
    ⊢ Nat.Primrec' fun v => f (g v) (h v)
  -/
  simpa using hf.comp' (hg.cons <| hh.cons Primrec'.nil)
  /-
    🎉 no goals
  -/


theorem prec' {n f g h} (hf : @Primrec' n f) (hg : @Primrec' n g) (hh : @Primrec' (n + 2) h) :
    @Primrec' n fun v => (f v).rec (g v) fun y IH : ℕ => h (y ::ᵥ IH ::ᵥ v) := by
  /-
    n : Nat
    f g : List.Vector Nat n → Nat
    h : List.Vector Nat (HAdd.hAdd n 2) → Nat
    hf : Nat.Primrec' f
    hg : Nat.Primrec' g
    hh : Nat.Primrec' h
    ⊢ Nat.Primrec' fun v => Nat.rec (g v) (fun y IH => h (List.Vector.cons y (List …
  -/
  simpa using comp' (prec hg hh) (hf.cons idv)
  /-
    🎉 no goals
  -/


theorem pred : @Primrec' 1 fun v => v.head.pred :=
                                                /-
                                                  v : List.Vector Nat (Nat.succ 0)
                                                  ⊢ Eq (Nat.rec 0 (fun y IH => (List.Vector.cons y (List.Vector.cons IH v)).head …
                                                -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
  (prec' head (const 0) head).of_eq fun v => by simp; cases v.head <;> rfl
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem add : @Primrec' 2 fun v => v.head + v.tail.head :=
  (prec head (succ.comp₁ _ (tail head))).of_eq fun v => by
    /-
      v : List.Vector Nat (HAdd.hAdd (Nat.succ 0) 1)
      ⊢ Eq (Nat.rec v.tail.head (fun y IH => (List.Vector.cons y (List.Vector.cons I …
    -/
                               /-
                                 🎉 no goals
                               -/
    simp; induction v.head <;> simp [*, Nat.succ_add]
                               /-
                                 🎉 no goals
                               -/


theorem sub : @Primrec' 2 fun v => v.head - v.tail.head := by
  have : @Primrec' 2 fun v ↦ (fun a b ↦ b - a) v.head v.tail.head := by
    refine (prec head (pred.comp₁ _ (tail head))).of_eq fun v => ?_
    simp; induction v.head <;> simp [*, Nat.sub_add_eq]
  /-
    this : Nat.Primrec' fun v => (fun a b => HSub.hSub b a) v.head v.tail.head
    ⊢ Nat.Primrec' fun v => HSub.hSub v.head v.tail.head
  -/
  simpa using comp₂ (fun a b => b - a) this (tail head) head
  /-
    🎉 no goals
  -/


theorem mul : @Primrec' 2 fun v => v.head * v.tail.head :=
  (prec (const 0) (tail (add.comp₂ _ (tail head) head))).of_eq fun v => by
    /-
      v : List.Vector Nat (HAdd.hAdd (Nat.succ 0) 1)
      ⊢ Eq (Nat.rec 0 (fun y IH => HAdd.hAdd (List.Vector.cons y (List.Vector.cons I …
    -/
                               /-
                                 🎉 no goals
                               -/
    simp; induction v.head <;> simp [*, Nat.succ_mul]; rw [add_comm]
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem if_lt {n a b f g} (ha : @Primrec' n a) (hb : @Primrec' n b) (hf : @Primrec' n f)
    (hg : @Primrec' n g) : @Primrec' n fun v => if a v < b v then f v else g v :=
  (prec' (sub.comp₂ _ hb ha) hg (tail <| tail hf)).of_eq fun v => by
    /-
      n : Nat
      a b f g : List.Vector Nat n → Nat
      ha : Nat.Primrec' a
      hb : Nat.Primrec' b
      hf : Nat.Primrec' f
      hg : Nat.Primrec' g
      v : List.Vector Nat n
      ⊢ Eq (Nat.rec (g v) (fun y IH => f (List.Vector.cons y (List.Vector.cons IH v) …
    -/
    cases e : b v - a v
      /-
        case zero
        n : Nat
        a b f g : List.Vector Nat n → Nat
        ha : Nat.Primrec' a
        hb : Nat.Primrec' b
        hf : Nat.Primrec' f
        hg : Nat.Primrec' g
        v : List.Vector Nat n
        e : Eq (HSub.hSub (b v) (a v)) 0
        ⊢ Eq (Nat.rec (g v) (fun y IH => f (List.Vector.cons y (List.Vector.cons IH v) …
      -/
    · simp [not_lt.2 (tsub_eq_zero_iff_le.mp e)]
      /-
        🎉 no goals
      -/
      /-
        case succ
        n : Nat
        a b f g : List.Vector Nat n → Nat
        ha : Nat.Primrec' a
        hb : Nat.Primrec' b
        hf : Nat.Primrec' f
        hg : Nat.Primrec' g
        v : List.Vector Nat n
        n✝ : Nat
        e : Eq (HSub.hSub (b v) (a v)) (HAdd.hAdd n✝ 1)
        ⊢ Eq (Nat.rec (g v) (fun y IH => f (List.Vector.cons y (List.Vector.cons IH v) …
      -/
    · simp [Nat.lt_of_sub_eq_succ e]
      /-
        🎉 no goals
      -/


theorem natPair : @Primrec' 2 fun v => v.head.pair v.tail.head :=
  if_lt head (tail head) (add.comp₂ _ (tail <| mul.comp₂ _ head head) head)
    (add.comp₂ _ (add.comp₂ _ (mul.comp₂ _ head head) head) (tail head))


protected theorem encode : ∀ {n}, @Primrec' n encode
                                     /-
                                       v : List.Vector Nat 0
                                       ⊢ Eq 0 (Encodable.encode v)
                                     -/
  | 0 => (const 0).of_eq fun v => by rw [v.eq_nil]; rfl
                                                    /-
                                                      🎉 no goals
                                                    -/
  | _ + 1 =>
    (succ.comp₁ _ (natPair.comp₂ _ head (tail Primrec'.encode))).of_eq fun ⟨_ :: _, _⟩ => rfl


theorem sqrt : @Primrec' 1 fun v => v.head.sqrt := by
  suffices H : ∀ n : ℕ, n.sqrt =
      n.rec 0 fun x y => if x.succ < y.succ * y.succ then y else y.succ by
    simp only [H, succ_eq_add_one]
    have :=
      @prec' 1 _ _
        (fun v => by
          have x := v.head; have y := v.tail.head
          exact if x.succ < y.succ * y.succ then y else y.succ)
        head (const 0) ?_
    · exact this
    have x1 : @Primrec' 3 fun v => v.head.succ := succ.comp₁ _ head
    have y1 : @Primrec' 3 fun v => v.tail.head.succ := succ.comp₁ _ (tail head)
    exact if_lt x1 (mul.comp₂ _ y1 y1) (tail head) y1
  /-
    ⊢ ∀ (n : Nat), Eq n.sqrt (Nat.rec 0 (fun x y => ite (LT.lt x.succ (HMul.hMul y …
  -/
  introv; symm
  /-
    n : Nat
    ⊢ Eq (Nat.rec 0 (fun x y => ite (LT.lt x.succ (HMul.hMul y.succ y.succ)) y y.s …
  -/
  induction' n with n IH; · simp
                            /-
                              🎉 no goals
                            -/
  /-
    case succ
    n : Nat
    IH : Eq (Nat.rec 0 (fun x y => ite (LT.lt x.succ (HMul.hMul y.succ y.succ)) y  …
    ⊢ Eq (Nat.rec 0 (fun x y => ite (LT.lt x.succ (HMul.hMul y.succ y.succ)) y y.s …
  -/
  dsimp; rw [IH]; split_ifs with h
    /-
      case pos
      n : Nat
      IH : Eq (Nat.rec 0 (fun x y => ite (LT.lt x.succ (HMul.hMul y.succ y.succ)) y  …
      h : LT.lt (HAdd.hAdd n 1) (HMul.hMul (HAdd.hAdd n.sqrt 1) (HAdd.hAdd n.sqrt 1))
      ⊢ Eq n.sqrt (HAdd.hAdd n 1).sqrt
    -/
  · exact le_antisymm (Nat.sqrt_le_sqrt (Nat.le_succ _)) (Nat.lt_succ_iff.1 <| Nat.sqrt_lt.2 h)
    /-
      🎉 no goals
    -/
  · exact
      Nat.eq_sqrt.2 ⟨not_lt.1 h, Nat.sqrt_lt.1 <| Nat.lt_succ_iff.2 <| Nat.sqrt_succ_le_succ_sqrt _⟩


theorem unpair₁ {n f} (hf : @Primrec' n f) : @Primrec' n fun v => (f v).unpair.1 := by
  /-
    n : Nat
    f : List.Vector Nat n → Nat
    hf : Nat.Primrec' f
    ⊢ Nat.Primrec' fun v => (Nat.unpair (f v)).1
  -/
  have s := sqrt.comp₁ _ hf
  /-
    n : Nat
    f : List.Vector Nat n → Nat
    hf : Nat.Primrec' f
    s : Nat.Primrec' fun v => (f v).sqrt
    ⊢ Nat.Primrec' fun v => (Nat.unpair (f v)).1
  -/
  have fss := sub.comp₂ _ hf (mul.comp₂ _ s s)
  /-
    n : Nat
    f : List.Vector Nat n → Nat
    hf : Nat.Primrec' f
    s : Nat.Primrec' fun v => (f v).sqrt
    fss : Nat.Primrec' fun v => HSub.hSub (f v) (HMul.hMul (f v).sqrt (f v).sqrt)
    ⊢ Nat.Primrec' fun v => (Nat.unpair (f v)).1
  -/
  refine (if_lt fss s fss s).of_eq fun v => ?_
  /-
    n : Nat
    f : List.Vector Nat n → Nat
    hf : Nat.Primrec' f
    s : Nat.Primrec' fun v => (f v).sqrt
    fss : Nat.Primrec' fun v => HSub.hSub (f v) (HMul.hMul (f v).sqrt (f v).sqrt)
    v : List.Vector Nat n
    ⊢ Eq (ite (LT.lt (HSub.hSub (f v) (HMul.hMul (f v).sqrt (f v).sqrt)) (f v).sqr …
  -/
                                   /-
                                     🎉 no goals
                                   -/
  simp [Nat.unpair]; split_ifs <;> rfl
                                   /-
                                     🎉 no goals
                                   -/


theorem unpair₂ {n f} (hf : @Primrec' n f) : @Primrec' n fun v => (f v).unpair.2 := by
  /-
    n : Nat
    f : List.Vector Nat n → Nat
    hf : Nat.Primrec' f
    ⊢ Nat.Primrec' fun v => (Nat.unpair (f v)).2
  -/
  have s := sqrt.comp₁ _ hf
  /-
    n : Nat
    f : List.Vector Nat n → Nat
    hf : Nat.Primrec' f
    s : Nat.Primrec' fun v => (f v).sqrt
    ⊢ Nat.Primrec' fun v => (Nat.unpair (f v)).2
  -/
  have fss := sub.comp₂ _ hf (mul.comp₂ _ s s)
  /-
    n : Nat
    f : List.Vector Nat n → Nat
    hf : Nat.Primrec' f
    s : Nat.Primrec' fun v => (f v).sqrt
    fss : Nat.Primrec' fun v => HSub.hSub (f v) (HMul.hMul (f v).sqrt (f v).sqrt)
    ⊢ Nat.Primrec' fun v => (Nat.unpair (f v)).2
  -/
  refine (if_lt fss s s (sub.comp₂ _ fss s)).of_eq fun v => ?_
  /-
    n : Nat
    f : List.Vector Nat n → Nat
    hf : Nat.Primrec' f
    s : Nat.Primrec' fun v => (f v).sqrt
    fss : Nat.Primrec' fun v => HSub.hSub (f v) (HMul.hMul (f v).sqrt (f v).sqrt)
    v : List.Vector Nat n
    ⊢ Eq (ite (LT.lt (HSub.hSub (f v) (HMul.hMul (f v).sqrt (f v).sqrt)) (f v).sqr …
  -/
                                   /-
                                     🎉 no goals
                                   -/
  simp [Nat.unpair]; split_ifs <;> rfl
                                   /-
                                     🎉 no goals
                                   -/


theorem of_prim {n f} : Primrec f → @Primrec' n f :=
  suffices ∀ f, Nat.Primrec f → @Primrec' 1 fun v => f v.head from fun hf =>
    (pred.comp₁ _ <|
          (this _ hf).comp₁ (fun m => Encodable.encode <| (@decode (List.Vector ℕ n) _ m).map f)
            Primrec'.encode).of_eq
                  /-
                    n : Nat
                    f : List.Vector Nat n → Nat
                    this : ∀ (f : Nat → Nat), Nat.Primrec f → Nat.Primrec' fun v => f v.head
                    hf : Primrec f
                    i : List.Vector Nat n
                    ⊢ Eq ((fun m => Encodable.encode (Option.map f (Encodable.decode m))) (Encodab …
                  -/
      fun i => by simp [encodek]
                  /-
                    🎉 no goals
                  -/
  fun f hf => by
  induction hf with
  | zero => exact const 0
  | succ => exact succ
  | left => exact unpair₁ head
  | right => exact unpair₂ head
  | pair _ _ hf hg => exact natPair.comp₂ _ hf hg
  | comp _ _ hf hg => exact hf.comp₁ _ hg
  | prec _ _ hf hg =>
    simpa using
      prec' (unpair₂ head) (hf.comp₁ _ (unpair₁ head))
        (hg.comp₁ _ <|
          natPair.comp₂ _ (unpair₁ <| tail <| tail head) (natPair.comp₂ _ head (tail head)))


theorem prim_iff {n f} : @Primrec' n f ↔ Primrec f :=
  ⟨to_prim, of_prim⟩


theorem prim_iff₁ {f : ℕ → ℕ} : (@Primrec' 1 fun v => f v.head) ↔ Primrec f :=
  prim_iff.trans
                                                                      /-
                                                                        f : Nat → Nat
                                                                        h : Primrec fun v => f v.head
                                                                        v : Nat
                                                                        ⊢ Eq (f (List.Vector.ofFn fun i => id v).head) (f v)
                                                                      -/
    ⟨fun h => (h.comp <| .vector_ofFn fun _ => .id).of_eq fun v => by simp, fun h =>
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
      h.comp .vector_head⟩


theorem prim_iff₂ {f : ℕ → ℕ → ℕ} : (@Primrec' 2 fun v => f v.head v.tail.head) ↔ Primrec₂ f :=
  prim_iff.trans
    ⟨fun h => (h.comp <| Primrec.vector_cons.comp .fst <|
                                                                    /-
                                                                      f : Nat → Nat → Nat
                                                                      h : Primrec fun v => f v.head v.tail.head
                                                                      v : Prod Nat Nat
                                                                      ⊢ Eq (f (List.Vector.cons v.1 (List.Vector.cons v.2 List.Vector.nil)).head (Li …
                                                                    -/
      Primrec.vector_cons.comp .snd (.const nil)).of_eq fun v => by simp,
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
    fun h => h.comp .vector_head (Primrec.vector_head.comp .vector_tail)⟩


theorem vec_iff {m n f} : @Vec m n f ↔ Primrec f :=
               /-
                 m n : Nat
                 f : List.Vector Nat m → List.Vector Nat n
                 h : Nat.Primrec'.Vec f
                 ⊢ Primrec f
               -/
  ⟨fun h => by simpa using Primrec.vector_ofFn fun i => to_prim (h i), fun h i =>
               /-
                 🎉 no goals
               -/
    of_prim <| Primrec.vector_get.comp h (.const i)⟩


theorem Primrec.nat_sqrt : Primrec Nat.sqrt :=
  Nat.Primrec'.prim_iff₁.1 Nat.Primrec'.sqrt

