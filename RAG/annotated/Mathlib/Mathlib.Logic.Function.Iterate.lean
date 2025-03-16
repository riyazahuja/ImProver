/-- Iterate a function. -/
def Nat.iterate {α : Sort u} (op : α → α) : ℕ → α → α
  | 0, a => a
  | succ k, a => iterate op k (op a)


@[inherit_doc Nat.iterate]
notation:max f "^["n"]" => Nat.iterate f n


@[simp]
theorem iterate_zero : f^[0] = id :=
  rfl


theorem iterate_zero_apply (x : α) : f^[0] x = x :=
  rfl


@[simp]
theorem iterate_succ (n : ℕ) : f^[n.succ] = f^[n] ∘ f :=
  rfl


theorem iterate_succ_apply (n : ℕ) (x : α) : f^[n.succ] x = f^[n] (f x) :=
  rfl


@[simp]
theorem iterate_id (n : ℕ) : (id : α → α)^[n] = id :=
                                 /-
                                   α : Type u
                                   n✝ n : Nat
                                   ihn : Eq (Nat.iterate id n) id
                                   ⊢ Eq (Nat.iterate id n.succ) id
                                 -/
  Nat.recOn n rfl fun n ihn ↦ by rw [iterate_succ, ihn, id_comp]
                                 /-
                                   🎉 no goals
                                 -/


theorem iterate_add (m : ℕ) : ∀ n : ℕ, f^[m + n] = f^[m] ∘ f^[n]
  | 0 => rfl
                     /-
                       α : Type u
                       f : α → α
                       m n : Nat
                       ⊢ Eq (Nat.iterate f (HAdd.hAdd m n.succ)) (Function.comp (Nat.iterate f m) (Na …
                     -/
  | Nat.succ n => by rw [Nat.add_succ, iterate_succ, iterate_succ, iterate_add m n]; rfl
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


theorem iterate_add_apply (m n : ℕ) (x : α) : f^[m + n] x = f^[m] (f^[n] x) := by
  /-
    α : Type u
    f : α → α
    m n : Nat
    x : α
    ⊢ Eq (Nat.iterate f (HAdd.hAdd m n) x) (Nat.iterate f m (Nat.iterate f n x))
  -/
  rw [iterate_add f m n]
  /-
    α : Type u
    f : α → α
    m n : Nat
    x : α
    ⊢ Eq (Function.comp (Nat.iterate f m) (Nat.iterate f n) x) (Nat.iterate f m (N …
  -/
  rfl
  /-
    🎉 no goals
  -/

-- can be proved by simp but this is shorter and more natural

@[simp high]
theorem iterate_one : f^[1] = f :=
  funext fun _ ↦ rfl


theorem iterate_mul (m : ℕ) : ∀ n, f^[m * n] = f^[m]^[n]
            /-
              α : Type u
              f : α → α
              m : Nat
              ⊢ Eq (Nat.iterate f (HMul.hMul m 0)) (Nat.iterate (Nat.iterate f m) 0)
            -/
  | 0 => by simp only [Nat.mul_zero, iterate_zero]
            /-
              🎉 no goals
            -/
                /-
                  α : Type u
                  f : α → α
                  m n : Nat
                  ⊢ Eq (Nat.iterate f (HMul.hMul m (HAdd.hAdd n 1))) (Nat.iterate (Nat.iterate f …
                -/
  | n + 1 => by simp only [Nat.mul_succ, Nat.mul_one, iterate_one, iterate_add, iterate_mul m n]
                /-
                  🎉 no goals
                -/


theorem iterate_fixed {x} (h : f x = x) (n : ℕ) : f^[n] x = x :=
                                 /-
                                   α : Type u
                                   f : α → α
                                   x : α
                                   h : Eq (f x) x
                                   n✝ n : Nat
                                   ihn : Eq (Nat.iterate f n x) x
                                   ⊢ Eq (Nat.iterate f n.succ x) x
                                 -/
  Nat.recOn n rfl fun n ihn ↦ by rw [iterate_succ_apply, h, ihn]
                                 /-
                                   🎉 no goals
                                 -/


theorem Injective.iterate (Hinj : Injective f) (n : ℕ) : Injective f^[n] :=
  Nat.recOn n injective_id fun _ ihn ↦ ihn.comp Hinj


theorem Surjective.iterate (Hsurj : Surjective f) (n : ℕ) : Surjective f^[n] :=
  Nat.recOn n surjective_id fun _ ihn ↦ ihn.comp Hsurj


theorem Bijective.iterate (Hbij : Bijective f) (n : ℕ) : Bijective f^[n] :=
  ⟨Hbij.1.iterate n, Hbij.2.iterate n⟩


theorem iterate_right {f : α → β} {ga : α → α} {gb : β → β} (h : Semiconj f ga gb) (n : ℕ) :
    Semiconj f ga^[n] gb^[n] :=
  Nat.recOn n id_right fun _ ihn ↦ ihn.comp_right h


theorem iterate_left {g : ℕ → α → α} (H : ∀ n, Semiconj f (g n) (g <| n + 1)) (n k : ℕ) :
    Semiconj f^[n] (g k) (g <| n + k) := by
  induction n generalizing k with
  | zero =>
    rw [Nat.zero_add]
    exact id_left
  | succ n ihn =>
    rw [Nat.add_right_comm, Nat.add_assoc]
    exact (H k).trans (ihn (k + 1))


theorem iterate_right (h : Commute f g) (n : ℕ) : Commute f g^[n] :=
  Semiconj.iterate_right h n


theorem iterate_left (h : Commute f g) (n : ℕ) : Commute f^[n] g :=
  (h.symm.iterate_right n).symm


theorem iterate_iterate (h : Commute f g) (m n : ℕ) : Commute f^[m] g^[n] :=
  (h.iterate_left m).iterate_right n


theorem iterate_eq_of_map_eq (h : Commute f g) (n : ℕ) {x} (hx : f x = g x) :
    f^[n] x = g^[n] x :=
  Nat.recOn n rfl fun n ihn ↦ by
    /-
      α : Type u
      f g : α → α
      h : Function.Commute f g
      n✝ : Nat
      x : α
      hx : Eq (f x) (g x)
      n : Nat
      ihn : Eq (Nat.iterate f n x) (Nat.iterate g n x)
      ⊢ Eq (Nat.iterate f n.succ x) (Nat.iterate g n.succ x)
    -/
    simp only [iterate_succ_apply, hx, (h.iterate_left n).eq, ihn, ((refl g).iterate_right n).eq]
    /-
      🎉 no goals
    -/


theorem comp_iterate (h : Commute f g) (n : ℕ) : (f ∘ g)^[n] = f^[n] ∘ g^[n] := by
  induction n with
  | zero => rfl
  | succ n ihn =>
    funext x
    simp only [ihn, (h.iterate_right n).eq, iterate_succ, comp_apply]


theorem iterate_self (n : ℕ) : Commute f^[n] f :=
  (refl f).iterate_left n


theorem self_iterate (n : ℕ) : Commute f f^[n] :=
  (refl f).iterate_right n


theorem iterate_iterate_self (m n : ℕ) : Commute f^[m] f^[n] :=
  (refl f).iterate_iterate m n


theorem Semiconj₂.iterate {f : α → α} {op : α → α → α} (hf : Semiconj₂ f op op) (n : ℕ) :
    Semiconj₂ f^[n] op op :=
  Nat.recOn n (Semiconj₂.id_left op) fun _ ihn ↦ ihn.comp hf


theorem iterate_succ' (n : ℕ) : f^[n.succ] = f ∘ f^[n] := by
  /-
    α : Type u
    f : α → α
    n : Nat
    ⊢ Eq (Nat.iterate f n.succ) (Function.comp f (Nat.iterate f n))
  -/
  rw [iterate_succ, (Commute.self_iterate f n).comp_eq]
  /-
    🎉 no goals
  -/


theorem iterate_succ_apply' (n : ℕ) (x : α) : f^[n.succ] x = f (f^[n] x) := by
  /-
    α : Type u
    f : α → α
    n : Nat
    x : α
    ⊢ Eq (Nat.iterate f n.succ x) (f (Nat.iterate f n x))
  -/
  rw [iterate_succ']
  /-
    α : Type u
    f : α → α
    n : Nat
    x : α
    ⊢ Eq (Function.comp f (Nat.iterate f n) x) (f (Nat.iterate f n x))
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem iterate_pred_comp_of_pos {n : ℕ} (hn : 0 < n) : f^[n.pred] ∘ f = f^[n] := by
  /-
    α : Type u
    f : α → α
    n : Nat
    hn : LT.lt 0 n
    ⊢ Eq (Function.comp (Nat.iterate f n.pred) f) (Nat.iterate f n)
  -/
  rw [← iterate_succ, Nat.succ_pred_eq_of_pos hn]
  /-
    🎉 no goals
  -/


theorem comp_iterate_pred_of_pos {n : ℕ} (hn : 0 < n) : f ∘ f^[n.pred] = f^[n] := by
  /-
    α : Type u
    f : α → α
    n : Nat
    hn : LT.lt 0 n
    ⊢ Eq (Function.comp f (Nat.iterate f n.pred)) (Nat.iterate f n)
  -/
  rw [← iterate_succ', Nat.succ_pred_eq_of_pos hn]
  /-
    🎉 no goals
  -/


/-- A recursor for the iterate of a function. -/
def Iterate.rec (p : α → Sort*) {f : α → α} (h : ∀ a, p a → p (f a)) {a : α} (ha : p a) (n : ℕ) :
    p (f^[n] a) :=
  match n with
  | 0 => ha
  | m+1 => Iterate.rec p h (h _ ha) m


theorem Iterate.rec_zero (p : α → Sort*) {f : α → α} (h : ∀ a, p a → p (f a)) {a : α} (ha : p a) :
    Iterate.rec p h ha 0 = ha :=
  rfl


theorem LeftInverse.iterate {g : α → α} (hg : LeftInverse g f) (n : ℕ) :
    LeftInverse g^[n] f^[n] :=
  Nat.recOn n (fun _ ↦ rfl) fun n ihn ↦ by
    /-
      α : Type u
      f g : α → α
      hg : Function.LeftInverse g f
      n✝ n : Nat
      ihn : Function.LeftInverse (Nat.iterate g n) (Nat.iterate f n)
      ⊢ Function.LeftInverse (Nat.iterate g n.succ) (Nat.iterate f n.succ)
    -/
    rw [iterate_succ', iterate_succ]
    /-
      α : Type u
      f g : α → α
      hg : Function.LeftInverse g f
      n✝ n : Nat
      ihn : Function.LeftInverse (Nat.iterate g n) (Nat.iterate f n)
      ⊢ Function.LeftInverse (Function.comp g (Nat.iterate g n)) (Function.comp (Nat …
    -/
    exact ihn.comp hg
    /-
      🎉 no goals
    -/


theorem RightInverse.iterate {g : α → α} (hg : RightInverse g f) (n : ℕ) :
    RightInverse g^[n] f^[n] :=
  LeftInverse.iterate hg n


theorem iterate_comm (f : α → α) (m n : ℕ) : f^[n]^[m] = f^[m]^[n] :=
                                               /-
                                                 α : Type u
                                                 f : α → α
                                                 m n : Nat
                                                 ⊢ Eq (Nat.iterate f (HMul.hMul n m)) (Nat.iterate f (HMul.hMul m n))
                                               -/
  (iterate_mul _ _ _).symm.trans (Eq.trans (by rw [Nat.mul_comm]) (iterate_mul _ _ _))
                                               /-
                                                 🎉 no goals
                                               -/


theorem iterate_commute (m n : ℕ) : Commute (fun f : α → α ↦ f^[m]) fun f ↦ f^[n] :=
  fun f ↦ iterate_comm f m n


lemma iterate_add_eq_iterate (hf : Injective f) : f^[m + n] a = f^[n] a ↔ f^[m] a = a :=
                /-
                  α : Type u
                  f : α → α
                  m n : Nat
                  a : α
                  hf : Function.Injective f
                  ⊢ Iff (Eq (Nat.iterate f (HAdd.hAdd m n) a) (Nat.iterate f n a)) (Eq (Nat.iter …
                -/
  Iff.trans (by rw [← iterate_add_apply, Nat.add_comm]) (hf.iterate n).eq_iff
                /-
                  🎉 no goals
                -/


alias ⟨iterate_cancel_of_add, _⟩ := iterate_add_eq_iterate


lemma iterate_cancel (hf : Injective f) (ha : f^[m] a = f^[n] a) : f^[m - n] a = a := by
  /-
    α : Type u
    f : α → α
    m n : Nat
    a : α
    hf : Function.Injective f
    ha : Eq (Nat.iterate f m a) (Nat.iterate f n a)
    ⊢ Eq (Nat.iterate f (HSub.hSub m n) a) a
  -/
  obtain h | h := Nat.le_total m n
  /-
    case inl
    α : Type u
    f : α → α
    m n : Nat
    a : α
    hf : Function.Injective f
    ha : Eq (Nat.iterate f m a) (Nat.iterate f n a)
    h : LE.le m n
    ⊢ Eq (Nat.iterate f (HSub.hSub m n) a) a
  -/
  { simp [Nat.sub_eq_zero_of_le h] }
  /-
    case inr
    α : Type u
    f : α → α
    m n : Nat
    a : α
    hf : Function.Injective f
    ha : Eq (Nat.iterate f m a) (Nat.iterate f n a)
    h : LE.le n m
    ⊢ Eq (Nat.iterate f (HSub.hSub m n) a) a
  -/
  { exact iterate_cancel_of_add hf (by rwa [Nat.sub_add_cancel h]) }
  /-
    🎉 no goals
  -/


theorem involutive_iff_iter_2_eq_id {α} {f : α → α} : Involutive f ↔ f^[2] = id :=
  funext_iff.symm


theorem foldl_const (f : α → α) (a : α) (l : List β) :
    l.foldl (fun b _ ↦ f b) a = f^[l.length] a := by
  induction l generalizing a with
  | nil => rfl
  | cons b l H => rw [length_cons, foldl, iterate_succ_apply, H]


theorem foldr_const (f : β → β) (b : β) : ∀ l : List α, l.foldr (fun _ ↦ f) b = f^[l.length] b
  | [] => rfl
                 /-
                   α : Type u
                   β : Type v
                   f : β → β
                   b : β
                   a : α
                   l : List α
                   ⊢ Eq (List.foldr (fun x => f) b (List.cons a l)) (Nat.iterate f (List.cons a l …
                 -/
  | a :: l => by rw [length_cons, foldr, foldr_const f b l, iterate_succ_apply']
                 /-
                   🎉 no goals
                 -/


