/-- `join S`, where `S` is a multiset of multisets, is the lift of the list join
  operation, that is, the union of all the sets.
     join {{1, 2}, {1, 2}, {0, 1}} = {0, 1, 1, 1, 2, 2} -/
def join : Multiset (Multiset α) → Multiset α :=
  sum


theorem coe_join : ∀ L : List (List α), join (L.map ((↑) : List α → Multiset α) :
    Multiset (Multiset α)) = L.flatten
  | [] => rfl
  | l :: L => by
      /-
        α : Type u_1
        l : List α
        L : List (List α)
        ⊢ Eq (↑(List.map Multiset.ofList (List.cons l L))).join ↑(List.cons l L).flatten
      -/
      exact congr_arg (fun s : Multiset α => ↑l + s) (coe_join L)
      /-
        🎉 no goals
      -/


@[simp]
theorem join_zero : @join α 0 = 0 :=
  rfl


@[simp]
theorem join_cons (s S) : @join α (s ::ₘ S) = s + join S :=
  sum_cons _ _


@[simp]
theorem join_add (S T) : @join α (S + T) = join S + join T :=
  sum_add _ _


@[simp]
theorem singleton_join (a) : join ({a} : Multiset (Multiset α)) = a :=
  sum_singleton _


@[simp]
theorem mem_join {a S} : a ∈ @join α S ↔ ∃ s ∈ S, a ∈ s :=
                              /-
                                α : Type u_1
                                a : α
                                S : Multiset (Multiset α)
                                ⊢ Iff (Membership.mem (Multiset.join 0) a) (Exists fun s => And (Membership.me …
                              -/
  Multiset.induction_on S (by simp) <| by
                              /-
                                🎉 no goals
                              -/
    /-
      α : Type u_1
      a : α
      S : Multiset (Multiset α)
      ⊢ ∀ (a_1 : Multiset α) (s : Multiset (Multiset α)), Iff (Membership.mem s.join …
    -/
    simp +contextual [or_and_right, exists_or]
    /-
      🎉 no goals
    -/


@[simp]
theorem card_join (S) : card (@join α S) = sum (map card S) :=
                              /-
                                α : Type u_1
                                S : Multiset (Multiset α)
                                ⊢ Eq (Multiset.join 0).card (Multiset.map Multiset.card 0).sum
                              -/
                              /-
                                🎉 no goals
                              -/
  Multiset.induction_on S (by simp) (by simp)
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
theorem map_join (f : α → β) (S : Multiset (Multiset α)) :
    map f (join S) = join (map (map f) S) := by
  induction S using Multiset.induction with
  | empty => simp
  | cons _ _ ih => simp [ih]


@[to_additive (attr := simp)]
theorem prod_join [CommMonoid α] {S : Multiset (Multiset α)} :
    prod (join S) = prod (map prod S) := by
  induction S using Multiset.induction with
  | empty => simp
  | cons _ _ ih => simp [ih]


theorem rel_join {r : α → β → Prop} {s t} (h : Rel (Rel r) s t) : Rel r s.join t.join := by
  induction h with
  | zero => simp
  | cons hab hst ih => simpa using hab.add ih


/-- `s.bind f` is the monad bind operation, defined as `(s.map f).join`. It is the union of `f a` as
`a` ranges over `s`. -/
def bind (s : Multiset α) (f : α → Multiset β) : Multiset β :=
  (s.map f).join


@[simp]
theorem coe_bind (l : List α) (f : α → List β) : (@bind α β l fun a => f a) = l.flatMap f := by
  /-
    α : Type u_1
    β : Type v
    l : List α
    f : α → List β
    ⊢ Eq ((↑l).bind fun a => ↑(f a)) ↑(l.flatMap f)
  -/
  rw [List.flatMap, ← coe_join, List.map_map]
  /-
    α : Type u_1
    β : Type v
    l : List α
    f : α → List β
    ⊢ Eq ((↑l).bind fun a => ↑(f a)) (↑(List.map (Function.comp Multiset.ofList f) …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem zero_bind : bind 0 f = 0 :=
  rfl


@[simp]
                                                            /-
                                                              α : Type u_1
                                                              β : Type v
                                                              a : α
                                                              s : Multiset α
                                                              f : α → Multiset β
                                                              ⊢ Eq ((Multiset.cons a s).bind f) (HAdd.hAdd (f a) (s.bind f))
                                                            -/
theorem cons_bind : (a ::ₘ s).bind f = f a + s.bind f := by simp [bind]
                                                            /-
                                                              🎉 no goals
                                                            -/


@[simp]
                                                /-
                                                  α : Type u_1
                                                  β : Type v
                                                  a : α
                                                  f : α → Multiset β
                                                  ⊢ Eq ((Singleton.singleton a).bind f) (f a)
                                                -/
theorem singleton_bind : bind {a} f = f a := by simp [bind]
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
                                                              /-
                                                                α : Type u_1
                                                                β : Type v
                                                                s t : Multiset α
                                                                f : α → Multiset β
                                                                ⊢ Eq ((HAdd.hAdd s t).bind f) (HAdd.hAdd (s.bind f) (t.bind f))
                                                              -/
theorem add_bind : (s + t).bind f = s.bind f + t.bind f := by simp [bind]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
                                                                   /-
                                                                     α : Type u_1
                                                                     β : Type v
                                                                     s : Multiset α
                                                                     ⊢ Eq (s.bind fun x => 0) 0
                                                                   -/
theorem bind_zero : s.bind (fun _ => 0 : α → Multiset β) = 0 := by simp [bind, join, nsmul_zero]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
                                                                           /-
                                                                             α : Type u_1
                                                                             β : Type v
                                                                             s : Multiset α
                                                                             f g : α → Multiset β
                                                                             ⊢ Eq (s.bind fun a => HAdd.hAdd (f a) (g a)) (HAdd.hAdd (s.bind f) (s.bind g))
                                                                           -/
theorem bind_add : (s.bind fun a => f a + g a) = s.bind f + s.bind g := by simp [bind, join]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[simp]
theorem bind_cons (f : α → β) (g : α → Multiset β) :
    (s.bind fun a => f a ::ₘ g a) = map f s + s.bind g :=
                              /-
                                α : Type u_1
                                β : Type v
                                s : Multiset α
                                f : α → β
                                g : α → Multiset β
                                ⊢ Eq (Multiset.bind 0 fun a => Multiset.cons (f a) (g a)) (HAdd.hAdd (Multiset …
                              -/
  Multiset.induction_on s (by simp)
                              /-
                                🎉 no goals
                              -/
        /-
          α : Type u_1
          β : Type v
          s : Multiset α
          f : α → β
          g : α → Multiset β
          ⊢ ∀ (a : α) (s : Multiset α), Eq (s.bind fun a => Multiset.cons (f a) (g a)) ( …
        -/
    (by simp +contextual [add_comm, add_left_comm, add_assoc])
        /-
          🎉 no goals
        -/


@[simp]
theorem bind_singleton (f : α → β) : (s.bind fun x => ({f x} : Multiset β)) = map f s :=
                              /-
                                α : Type u_1
                                β : Type v
                                s : Multiset α
                                f : α → β
                                ⊢ Eq (Multiset.bind 0 fun x => Singleton.singleton (f x)) (Multiset.map f 0)
                              -/
                              /-
                                🎉 no goals
                              -/
  Multiset.induction_on s (by rw [zero_bind, map_zero]) (by simp [singleton_add])
                                                            /-
                                                              🎉 no goals
                                                            -/


@[simp]
theorem mem_bind {b s} {f : α → Multiset β} : b ∈ bind s f ↔ ∃ a ∈ s, b ∈ f a := by
  /-
    α : Type u_1
    β : Type v
    b : β
    s : Multiset α
    f : α → Multiset β
    ⊢ Iff (Membership.mem (s.bind f) b) (Exists fun a => And (Membership.mem s a)  …
  -/
  simp [bind]
  /-
    🎉 no goals
  -/


@[simp]
                                                                   /-
                                                                     α : Type u_1
                                                                     β : Type v
                                                                     s : Multiset α
                                                                     f : α → Multiset β
                                                                     ⊢ Eq (s.bind f).card (Multiset.map (Function.comp Multiset.card f) s).sum
                                                                   -/
theorem card_bind : card (s.bind f) = (s.map (card ∘ f)).sum := by simp [bind]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


theorem bind_congr {f g : α → Multiset β} {m : Multiset α} :
                                                     /-
                                                       α : Type u_1
                                                       β : Type v
                                                       f g : α → Multiset β
                                                       m : Multiset α
                                                       ⊢ (∀ (a : α), Membership.mem m a → Eq (f a) (g a)) → Eq (m.bind f) (m.bind g)
                                                     -/
    (∀ a ∈ m, f a = g a) → bind m f = bind m g := by simp +contextual [bind]
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem bind_hcongr {β' : Type v} {m : Multiset α} {f : α → Multiset β} {f' : α → Multiset β'}
    (h : β = β') (hf : ∀ a ∈ m, HEq (f a) (f' a)) : HEq (bind m f) (bind m f') := by
  /-
    α : Type u_1
    β β' : Type v
    m : Multiset α
    f : α → Multiset β
    f' : α → Multiset β'
    h : Eq β β'
    hf : ∀ (a : α), Membership.mem m a → HEq (f a) (f' a)
    ⊢ HEq (m.bind f) (m.bind f')
  -/
  subst h
  /-
    α : Type u_1
    β : Type v
    m : Multiset α
    f f' : α → Multiset β
    hf : ∀ (a : α), Membership.mem m a → HEq (f a) (f' a)
    ⊢ HEq (m.bind f) (m.bind f')
  -/
  simp only [heq_eq_eq] at hf
  /-
    α : Type u_1
    β : Type v
    m : Multiset α
    f f' : α → Multiset β
    hf : ∀ (a : α), Membership.mem m a → Eq (f a) (f' a)
    ⊢ HEq (m.bind f) (m.bind f')
  -/
  simp [bind_congr hf]
  /-
    🎉 no goals
  -/


theorem map_bind (m : Multiset α) (n : α → Multiset β) (f : β → γ) :
                                                         /-
                                                           α : Type u_1
                                                           β : Type v
                                                           γ : Type u_2
                                                           m : Multiset α
                                                           n : α → Multiset β
                                                           f : β → γ
                                                           ⊢ Eq (Multiset.map f (m.bind n)) (m.bind fun a => Multiset.map f (n a))
                                                         -/
    map f (bind m n) = bind m fun a => map f (n a) := by simp [bind]
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem bind_map (m : Multiset α) (n : β → Multiset γ) (f : α → β) :
    bind (map f m) n = bind m fun a => n (f a) :=
                              /-
                                α : Type u_1
                                β : Type v
                                γ : Type u_2
                                m : Multiset α
                                n : β → Multiset γ
                                f : α → β
                                ⊢ Eq ((Multiset.map f 0).bind n) (Multiset.bind 0 fun a => n (f a))
                              -/
                              /-
                                🎉 no goals
                              -/
  Multiset.induction_on m (by simp) (by simp +contextual)
                                        /-
                                          🎉 no goals
                                        -/


theorem bind_assoc {s : Multiset α} {f : α → Multiset β} {g : β → Multiset γ} :
    (s.bind f).bind g = s.bind fun a => (f a).bind g :=
                              /-
                                α : Type u_1
                                β : Type v
                                γ : Type u_2
                                s : Multiset α
                                f : α → Multiset β
                                g : β → Multiset γ
                                ⊢ Eq ((Multiset.bind 0 f).bind g) (Multiset.bind 0 fun a => (f a).bind g)
                              -/
                              /-
                                🎉 no goals
                              -/
  Multiset.induction_on s (by simp) (by simp +contextual)
                                        /-
                                          🎉 no goals
                                        -/


theorem bind_bind (m : Multiset α) (n : Multiset β) {f : α → β → Multiset γ} :
    ((bind m) fun a => (bind n) fun b => f a b) = (bind n) fun b => (bind m) fun a => f a b :=
                              /-
                                α : Type u_1
                                β : Type v
                                γ : Type u_2
                                m : Multiset α
                                n : Multiset β
                                f : α → β → Multiset γ
                                ⊢ Eq (Multiset.bind 0 fun a => n.bind fun b => f a b) (n.bind fun b => Multise …
                              -/
                              /-
                                🎉 no goals
                              -/
  Multiset.induction_on m (by simp) (by simp +contextual)
                                        /-
                                          🎉 no goals
                                        -/


theorem bind_map_comm (m : Multiset α) (n : Multiset β) {f : α → β → γ} :
    ((bind m) fun a => n.map fun b => f a b) = (bind n) fun b => m.map fun a => f a b :=
                              /-
                                α : Type u_1
                                β : Type v
                                γ : Type u_2
                                m : Multiset α
                                n : Multiset β
                                f : α → β → γ
                                ⊢ Eq (Multiset.bind 0 fun a => Multiset.map (fun b => f a b) n) (n.bind fun b  …
                              -/
                              /-
                                🎉 no goals
                              -/
  Multiset.induction_on m (by simp) (by simp +contextual)
                                        /-
                                          🎉 no goals
                                        -/


@[to_additive (attr := simp)]
theorem prod_bind [CommMonoid β] (s : Multiset α) (t : α → Multiset β) :
                                                             /-
                                                               α : Type u_1
                                                               β : Type v
                                                               inst✝ : CommMonoid β
                                                               s : Multiset α
                                                               t : α → Multiset β
                                                               ⊢ Eq (s.bind t).prod (Multiset.map (fun a => (t a).prod) s).prod
                                                             -/
    (s.bind t).prod = (s.map fun a => (t a).prod).prod := by simp [bind]
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem rel_bind {r : α → β → Prop} {p : γ → δ → Prop} {s t} {f : α → Multiset γ}
    {g : β → Multiset δ} (h : (r ⇒ Rel p) f g) (hst : Rel r s t) :
    Rel p (s.bind f) (t.bind g) := by
  /-
    α : Type u_1
    β : Type v
    γ : Type u_2
    δ : Type u_3
    r : α → β → Prop
    p : γ → δ → Prop
    s : Multiset α
    t : Multiset β
    f : α → Multiset γ
    g : β → Multiset δ
    h : Relator.LiftFun r (Multiset.Rel p) f g
    hst : Multiset.Rel r s t
    ⊢ Multiset.Rel p (s.bind f) (t.bind g)
  -/
  apply rel_join
  /-
    case h
    α : Type u_1
    β : Type v
    γ : Type u_2
    δ : Type u_3
    r : α → β → Prop
    p : γ → δ → Prop
    s : Multiset α
    t : Multiset β
    f : α → Multiset γ
    g : β → Multiset δ
    h : Relator.LiftFun r (Multiset.Rel p) f g
    hst : Multiset.Rel r s t
    ⊢ Multiset.Rel (Multiset.Rel p) (Multiset.map f s) (Multiset.map g t)
  -/
  rw [rel_map]
  /-
    case h
    α : Type u_1
    β : Type v
    γ : Type u_2
    δ : Type u_3
    r : α → β → Prop
    p : γ → δ → Prop
    s : Multiset α
    t : Multiset β
    f : α → Multiset γ
    g : β → Multiset δ
    h : Relator.LiftFun r (Multiset.Rel p) f g
    hst : Multiset.Rel r s t
    ⊢ Multiset.Rel (fun a b => Multiset.Rel p (f a) (g b)) s t
  -/
  exact hst.mono fun a _ b _ hr => h hr
  /-
    🎉 no goals
  -/


theorem count_sum [DecidableEq α] {m : Multiset β} {f : β → Multiset α} {a : α} :
    count a (map f m).sum = sum (m.map fun b => count a <| f b) :=
                              /-
                                α : Type u_1
                                β : Type v
                                inst✝ : DecidableEq α
                                m : Multiset β
                                f : β → Multiset α
                                a : α
                                ⊢ Eq (Multiset.count a (Multiset.map f 0).sum) (Multiset.map (fun b => Multise …
                              -/
                              /-
                                🎉 no goals
                              -/
  Multiset.induction_on m (by simp) (by simp)
                                        /-
                                          🎉 no goals
                                        -/


theorem count_bind [DecidableEq α] {m : Multiset β} {f : β → Multiset α} {a : α} :
    count a (bind m f) = sum (m.map fun b => count a <| f b) :=
  count_sum


theorem le_bind {α β : Type*} {f : α → Multiset β} (S : Multiset α) {x : α} (hx : x ∈ S) :
    f x ≤ S.bind f := by
  classical
  refine le_iff_count.2 fun a ↦ ?_
  obtain ⟨m', hm'⟩ := exists_cons_of_mem <| mem_map_of_mem (fun b ↦ count a (f b)) hx
  rw [count_bind, hm', sum_cons]
  exact Nat.le_add_right _ _

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): @[simp] removed because not in normal form

theorem attach_bind_coe (s : Multiset α) (f : α → Multiset β) :
    (s.attach.bind fun i => f i) = s.bind f :=
  congr_arg join <| attach_map_val' _ _


@[simp] lemma nodup_bind :
    Nodup (bind s f) ↔ (∀ a ∈ s, Nodup (f a)) ∧ s.Pairwise (Disjoint on f) := by
  /-
    α : Type u_1
    β : Type v
    s : Multiset α
    f : α → Multiset β
    ⊢ Iff (s.bind f).Nodup (And (∀ (a : α), Membership.mem s a → (f a).Nodup) (Mul …
  -/
  have : ∀ a, ∃ l : List β, f a = l := fun a => Quot.induction_on (f a) fun l => ⟨l, rfl⟩
  /-
    α : Type u_1
    β : Type v
    s : Multiset α
    f : α → Multiset β
    this : ∀ (a : α), Exists fun l => Eq (f a) ↑l
    ⊢ Iff (s.bind f).Nodup (And (∀ (a : α), Membership.mem s a → (f a).Nodup) (Mul …
  -/
  choose f' h' using this
  /-
    α : Type u_1
    β : Type v
    s : Multiset α
    f : α → Multiset β
    f' : α → List β
    h' : ∀ (a : α), Eq (f a) ↑(f' a)
    ⊢ Iff (s.bind f).Nodup (And (∀ (a : α), Membership.mem s a → (f a).Nodup) (Mul …
  -/
  have : f = fun a ↦ ofList (f' a) := funext h'
  /-
    α : Type u_1
    β : Type v
    s : Multiset α
    f : α → Multiset β
    f' : α → List β
    h' : ∀ (a : α), Eq (f a) ↑(f' a)
    this : Eq f fun a => ↑(f' a)
    ⊢ Iff (s.bind f).Nodup (And (∀ (a : α), Membership.mem s a → (f a).Nodup) (Mul …
  -/
  have hd : Symmetric fun a b ↦ List.Disjoint (f' a) (f' b) := fun a b h ↦ h.symm
  exact Quot.induction_on s <| by
    unfold Function.onFun
    simp [this, List.nodup_flatMap, pairwise_coe_iff_pairwise hd]


@[simp]
lemma dedup_bind_dedup [DecidableEq α] [DecidableEq β] (s : Multiset α) (f : α → Multiset β) :
    (s.dedup.bind f).dedup = (s.bind f).dedup := by
  /-
    α : Type u_1
    β : Type v
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    s : Multiset α
    f : α → Multiset β
    ⊢ Eq (s.dedup.bind f).dedup (s.bind f).dedup
  -/
  ext x
  -- Porting note: was `simp_rw [count_dedup, mem_bind, mem_dedup]`
  /-
    case a
    α : Type u_1
    β : Type v
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    s : Multiset α
    f : α → Multiset β
    x : β
    ⊢ Eq (Multiset.count x (s.dedup.bind f).dedup) (Multiset.count x (s.bind f).de …
  -/
  simp_rw [count_dedup]
  /-
    case a
    α : Type u_1
    β : Type v
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    s : Multiset α
    f : α → Multiset β
    x : β
    ⊢ Eq (ite (Membership.mem (s.dedup.bind f) x) 1 0) (ite (Membership.mem (s.bin …
  -/
  refine if_congr ?_ rfl rfl
  /-
    case a
    α : Type u_1
    β : Type v
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    s : Multiset α
    f : α → Multiset β
    x : β
    ⊢ Iff (Membership.mem (s.dedup.bind f) x) (Membership.mem (s.bind f) x)
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The multiplicity of `(a, b)` in `s ×ˢ t` is
  the product of the multiplicity of `a` in `s` and `b` in `t`. -/
def product (s : Multiset α) (t : Multiset β) : Multiset (α × β) :=
  s.bind fun a => t.map <| Prod.mk a


instance instSProd : SProd (Multiset α) (Multiset β) (Multiset (α × β)) where
  sprod := Multiset.product


@[simp]
theorem coe_product (l₁ : List α) (l₂ : List β) :
    (l₁ : Multiset α) ×ˢ (l₂ : Multiset β) = (l₁ ×ˢ l₂) := by
  /-
    α : Type u_1
    β : Type v
    l₁ : List α
    l₂ : List β
    ⊢ Eq (SProd.sprod ↑l₁ ↑l₂) ↑(SProd.sprod l₁ l₂)
  -/
  dsimp only [SProd.sprod]
  /-
    α : Type u_1
    β : Type v
    l₁ : List α
    l₂ : List β
    ⊢ Eq ((↑l₁).product ↑l₂) ↑(l₁.product l₂)
  -/
  rw [product, List.product, ← coe_bind]
  /-
    α : Type u_1
    β : Type v
    l₁ : List α
    l₂ : List β
    ⊢ Eq ((↑l₁).bind fun a => Multiset.map (Prod.mk a) ↑l₂) ((↑l₁).bind fun a => ↑ …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem zero_product : (0 : Multiset α) ×ˢ t = 0 :=
  rfl


@[simp]
                                                                         /-
                                                                           α : Type u_1
                                                                           β : Type v
                                                                           a : α
                                                                           s : Multiset α
                                                                           t : Multiset β
                                                                           ⊢ Eq (SProd.sprod (Multiset.cons a s) t) (HAdd.hAdd (Multiset.map (Prod.mk a)  …
                                                                         -/
theorem cons_product : (a ::ₘ s) ×ˢ t = map (Prod.mk a) t + s ×ˢ t := by simp [SProd.sprod, product]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[simp]
                                                       /-
                                                         α : Type u_1
                                                         β : Type v
                                                         s : Multiset α
                                                         ⊢ Eq (SProd.sprod s 0) 0
                                                       -/
theorem product_zero : s ×ˢ (0 : Multiset β) = 0 := by simp [SProd.sprod, product]
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp]
theorem product_cons : s ×ˢ (b ::ₘ t) = (s.map fun a => (a, b)) + s ×ˢ t := by
  /-
    α : Type u_1
    β : Type v
    b : β
    s : Multiset α
    t : Multiset β
    ⊢ Eq (SProd.sprod s (Multiset.cons b t)) (HAdd.hAdd (Multiset.map (fun a => {  …
  -/
  simp [SProd.sprod, product]
  /-
    🎉 no goals
  -/


@[simp]
theorem product_singleton : ({a} : Multiset α) ×ˢ ({b} : Multiset β) = {(a, b)} := by
  /-
    α : Type u_1
    β : Type v
    a : α
    b : β
    ⊢ Eq (SProd.sprod (Singleton.singleton a) (Singleton.singleton b)) (Singleton. …
  -/
  simp only [SProd.sprod, product, bind_singleton, map_singleton]
  /-
    🎉 no goals
  -/


@[simp]
theorem add_product (s t : Multiset α) (u : Multiset β) : (s + t) ×ˢ u = s ×ˢ u + t ×ˢ u := by
  /-
    α : Type u_1
    β : Type v
    s t : Multiset α
    u : Multiset β
    ⊢ Eq (SProd.sprod (HAdd.hAdd s t) u) (HAdd.hAdd (SProd.sprod s u) (SProd.sprod …
  -/
  simp [SProd.sprod, product]
  /-
    🎉 no goals
  -/


@[simp]
theorem product_add (s : Multiset α) : ∀ t u : Multiset β, s ×ˢ (t + u) = s ×ˢ t + s ×ˢ u :=
  Multiset.induction_on s (fun _ _ => rfl) fun a s IH t u => by
    /-
      α : Type u_1
      β : Type v
      s✝ : Multiset α
      a : α
      s : Multiset α
      IH : ∀ (t u : Multiset β), Eq (SProd.sprod s (HAdd.hAdd t u)) (HAdd.hAdd (SPro …
      t u : Multiset β
      ⊢ Eq (SProd.sprod (Multiset.cons a s) (HAdd.hAdd t u)) (HAdd.hAdd (SProd.sprod …
    -/
    rw [cons_product, IH]
    /-
      α : Type u_1
      β : Type v
      s✝ : Multiset α
      a : α
      s : Multiset α
      IH : ∀ (t u : Multiset β), Eq (SProd.sprod s (HAdd.hAdd t u)) (HAdd.hAdd (SPro …
      t u : Multiset β
      ⊢ Eq (HAdd.hAdd (Multiset.map (Prod.mk a) (HAdd.hAdd t u)) (HAdd.hAdd (SProd.s …
    -/
    simp [add_comm, add_left_comm, add_assoc]
    /-
      🎉 no goals
    -/


@[simp]
                                                             /-
                                                               α : Type u_1
                                                               β : Type v
                                                               s : Multiset α
                                                               t : Multiset β
                                                               ⊢ Eq (SProd.sprod s t).card (HMul.hMul s.card t.card)
                                                             -/
theorem card_product : card (s ×ˢ t) = card s * card t := by simp [SProd.sprod, product]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp] lemma mem_product : ∀ {p : α × β}, p ∈ @product α β s t ↔ p.1 ∈ s ∧ p.2 ∈ t
                 /-
                   α : Type u_1
                   β : Type v
                   s : Multiset α
                   t : Multiset β
                   a : α
                   b : β
                   ⊢ Iff (Membership.mem (s.product t) { fst := a, snd := b }) (And (Membership.m …
                 -/
  | (a, b) => by simp [product, and_left_comm]
                 /-
                   🎉 no goals
                 -/


protected theorem Nodup.product : Nodup s → Nodup t → Nodup (s ×ˢ t) :=
                                                  /-
                                                    α : Type u_1
                                                    β : Type v
                                                    s : Multiset α
                                                    t : Multiset β
                                                    l₁ : List α
                                                    l₂ : List β
                                                    d₁ : Multiset.Nodup (Quotient.mk (List.isSetoid α) l₁)
                                                    d₂ : Multiset.Nodup (Quotient.mk (List.isSetoid β) l₂)
                                                    ⊢ (SProd.sprod (Quotient.mk (List.isSetoid α) l₁) (Quotient.mk (List.isSetoid  …
                                                  -/
  Quotient.inductionOn₂ s t fun l₁ l₂ d₁ d₂ => by simp [List.Nodup.product d₁ d₂]
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- `Multiset.sigma s t` is the dependent version of `Multiset.product`. It is the sum of
  `(a, b)` as `a` ranges over `s` and `b` ranges over `t a`. -/
protected def sigma (s : Multiset α) (t : ∀ a, Multiset (σ a)) : Multiset (Σa, σ a) :=
  s.bind fun a => (t a).map <| Sigma.mk a


@[simp]
theorem coe_sigma (l₁ : List α) (l₂ : ∀ a, List (σ a)) :
    (@Multiset.sigma α σ l₁ fun a => l₂ a) = l₁.sigma l₂ := by
  /-
    α : Type u_1
    σ : α → Type u_4
    l₁ : List α
    l₂ : (a : α) → List (σ a)
    ⊢ Eq ((↑l₁).sigma fun a => ↑(l₂ a)) ↑(l₁.sigma l₂)
  -/
  rw [Multiset.sigma, List.sigma, ← coe_bind]
  /-
    α : Type u_1
    σ : α → Type u_4
    l₁ : List α
    l₂ : (a : α) → List (σ a)
    ⊢ Eq ((↑l₁).bind fun a => Multiset.map (Sigma.mk a) ↑(l₂ a)) ((↑l₁).bind fun a …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem zero_sigma : @Multiset.sigma α σ 0 t = 0 :=
  rfl


@[simp]
theorem cons_sigma : (a ::ₘ s).sigma t = (t a).map (Sigma.mk a) + s.sigma t := by
  /-
    α : Type u_1
    σ : α → Type u_4
    a : α
    s : Multiset α
    t : (a : α) → Multiset (σ a)
    ⊢ Eq ((Multiset.cons a s).sigma t) (HAdd.hAdd (Multiset.map (Sigma.mk a) (t a) …
  -/
  simp [Multiset.sigma]
  /-
    🎉 no goals
  -/


@[simp]
theorem sigma_singleton (b : α → β) :
    (({a} : Multiset α).sigma fun a => ({b a} : Multiset β)) = {⟨a, b a⟩} :=
  rfl


@[simp]
theorem add_sigma (s t : Multiset α) (u : ∀ a, Multiset (σ a)) :
                                                  /-
                                                    α : Type u_1
                                                    σ : α → Type u_4
                                                    s t : Multiset α
                                                    u : (a : α) → Multiset (σ a)
                                                    ⊢ Eq ((HAdd.hAdd s t).sigma u) (HAdd.hAdd (s.sigma u) (t.sigma u))
                                                  -/
    (s + t).sigma u = s.sigma u + t.sigma u := by simp [Multiset.sigma]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
theorem sigma_add :
    ∀ t u : ∀ a, Multiset (σ a), (s.sigma fun a => t a + u a) = s.sigma t + s.sigma u :=
  Multiset.induction_on s (fun _ _ => rfl) fun a s IH t u => by
    /-
      α : Type u_1
      σ : α → Type u_4
      s✝ : Multiset α
      a : α
      s : Multiset α
      IH : ∀ (t u : (a : α) → Multiset (σ a)), Eq (s.sigma fun a => HAdd.hAdd (t a)  …
      t u : (a : α) → Multiset (σ a)
      ⊢ Eq ((Multiset.cons a s).sigma fun a => HAdd.hAdd (t a) (u a)) (HAdd.hAdd ((M …
    -/
    rw [cons_sigma, IH]
    /-
      α : Type u_1
      σ : α → Type u_4
      s✝ : Multiset α
      a : α
      s : Multiset α
      IH : ∀ (t u : (a : α) → Multiset (σ a)), Eq (s.sigma fun a => HAdd.hAdd (t a)  …
      t u : (a : α) → Multiset (σ a)
      ⊢ Eq (HAdd.hAdd (Multiset.map (Sigma.mk a) (HAdd.hAdd (t a) (u a))) (HAdd.hAdd …
    -/
    simp [add_comm, add_left_comm, add_assoc]
    /-
      🎉 no goals
    -/


@[simp]
theorem card_sigma : card (s.sigma t) = sum (map (fun a => card (t a)) s) := by
  /-
    α : Type u_1
    σ : α → Type u_4
    s : Multiset α
    t : (a : α) → Multiset (σ a)
    ⊢ Eq (s.sigma t).card (Multiset.map (fun a => (t a).card) s).sum
  -/
  simp [Multiset.sigma, (· ∘ ·)]
  /-
    🎉 no goals
  -/


@[simp] lemma mem_sigma : ∀ {p : Σa, σ a}, p ∈ @Multiset.sigma α σ s t ↔ p.1 ∈ s ∧ p.2 ∈ t p.1
                 /-
                   α : Type u_1
                   σ : α → Type u_4
                   s : Multiset α
                   t : (a : α) → Multiset (σ a)
                   a : α
                   b : σ a
                   ⊢ Iff (Membership.mem (s.sigma t) ⟨a, b⟩) (And (Membership.mem s ⟨a, b⟩.fst) ( …
                 -/
  | ⟨a, b⟩ => by simp [Multiset.sigma, and_assoc, and_left_comm]
                 /-
                   🎉 no goals
                 -/


protected theorem Nodup.sigma {σ : α → Type*} {t : ∀ a, Multiset (σ a)} :
    Nodup s → (∀ a, Nodup (t a)) → Nodup (s.sigma t) :=
  Quot.induction_on s fun l₁ => by
    /-
      α : Type u_1
      s : Multiset α
      σ : α → Type u_5
      t : (a : α) → Multiset (σ a)
      l₁ : List α
      ⊢ Multiset.Nodup (Quot.mk (⇑(List.isSetoid α)) l₁) → (∀ (a : α), (t a).Nodup)  …
    -/
    choose f hf using fun a => Quotient.exists_rep (t a)
    /-
      α : Type u_1
      s : Multiset α
      σ : α → Type u_5
      t : (a : α) → Multiset (σ a)
      l₁ : List α
      f : (a : α) → List (σ a)
      hf : ∀ (a : α), Eq (Quotient.mk (List.isSetoid (σ a)) (f a)) (t a)
      ⊢ Multiset.Nodup (Quot.mk (⇑(List.isSetoid α)) l₁) → (∀ (a : α), (t a).Nodup)  …
    -/
    simpa [← funext hf] using List.Nodup.sigma
    /-
      🎉 no goals
    -/


