theorem coe_def : (fun a ↦ ↑a : α → Option α) = some :=
  rfl


                                                                                          /-
                                                                                            α : Type u_1
                                                                                            β : Type u_2
                                                                                            f : α → β
                                                                                            y : β
                                                                                            o : Option α
                                                                                            ⊢ Iff (Membership.mem (Option.map f o) y) (Exists fun x => And (Membership.mem …
                                                                                          -/
theorem mem_map {f : α → β} {y : β} {o : Option α} : y ∈ o.map f ↔ ∃ x ∈ o, f x = y := by simp
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/

-- The simpNF linter says that the LHS can be simplified via `Option.mem_def`.
-- However this is a higher priority lemma.
-- https://github.com/leanprover/std4/issues/207

@[simp 1100, nolint simpNF]
theorem mem_map_of_injective {f : α → β} (H : Function.Injective f) {a : α} {o : Option α} :
    f a ∈ o.map f ↔ a ∈ o := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    H : Function.Injective f
    a : α
    o : Option α
    ⊢ Iff (Membership.mem (Option.map f o) (f a)) (Membership.mem o a)
  -/
  aesop
  /-
    🎉 no goals
  -/


theorem forall_mem_map {f : α → β} {o : Option α} {p : β → Prop} :
                                                  /-
                                                    α : Type u_1
                                                    β : Type u_2
                                                    f : α → β
                                                    o : Option α
                                                    p : β → Prop
                                                    ⊢ Iff (∀ (y : β), Membership.mem (Option.map f o) y → p y) (∀ (x : α), Members …
                                                  -/
    (∀ y ∈ o.map f, p y) ↔ ∀ x ∈ o, p (f x) := by simp
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem exists_mem_map {f : α → β} {o : Option α} {p : β → Prop} :
                                                  /-
                                                    α : Type u_1
                                                    β : Type u_2
                                                    f : α → β
                                                    o : Option α
                                                    p : β → Prop
                                                    ⊢ Iff (Exists fun y => And (Membership.mem (Option.map f o) y) (p y)) (Exists  …
                                                  -/
    (∃ y ∈ o.map f, p y) ↔ ∃ x ∈ o, p (f x) := by simp
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem coe_get {o : Option α} (h : o.isSome) : ((Option.get _ h : α) : Option α) = o :=
  Option.some_get h


theorem eq_of_mem_of_mem {a : α} {o1 o2 : Option α} (h1 : a ∈ o1) (h2 : a ∈ o2) : o1 = o2 :=
  h1.trans h2.symm


theorem Mem.leftUnique : Relator.LeftUnique ((· ∈ ·) : α → Option α → Prop) :=
  fun _ _ _=> mem_unique


theorem some_injective (α : Type*) : Function.Injective (@some α) := fun _ _ ↦ some_inj.mp


/-- `Option.map f` is injective if `f` is injective. -/
theorem map_injective {f : α → β} (Hf : Function.Injective f) : Function.Injective (Option.map f)
  | none, none, _ => rfl
                              /-
                                α : Type u_1
                                β : Type u_2
                                f : α → β
                                Hf : Function.Injective f
                                a₁ a₂ : α
                                H : Eq (Option.map f (Option.some a₁)) (Option.map f (Option.some a₂))
                                ⊢ Eq (Option.some a₁) (Option.some a₂)
                              -/
  | some a₁, some a₂, H => by rw [Hf (Option.some.inj H)]
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem map_comp_some (f : α → β) : Option.map f ∘ some = some ∘ f :=
  rfl


@[simp]
theorem none_bind' (f : α → Option β) : none.bind f = none :=
  rfl


@[simp]
theorem some_bind' (a : α) (f : α → Option β) : (some a).bind f = f a :=
  rfl


theorem bind_eq_some' {x : Option α} {f : α → Option β} {b : β} :
    x.bind f = some b ↔ ∃ a, x = some a ∧ f a = some b := by
  /-
    α : Type u_1
    β : Type u_2
    x : Option α
    f : α → Option β
    b : β
    ⊢ Iff (Eq (x.bind f) (Option.some b)) (Exists fun a => And (Eq x (Option.some  …
  -/
              /-
                🎉 no goals
              -/
  cases x <;> simp
              /-
                🎉 no goals
              -/


theorem bind_congr {f g : α → Option β} {x : Option α}
    (h : ∀ a ∈ x, f a = g a) : x.bind f = x.bind g := by
  /-
    α : Type u_1
    β : Type u_2
    f g : α → Option β
    x : Option α
    h : ∀ (a : α), Membership.mem x a → Eq (f a) (g a)
    ⊢ Eq (x.bind f) (x.bind g)
  -/
              /-
                🎉 no goals
              -/
  cases x <;> simp only [some_bind, none_bind, mem_def, h]
              /-
                🎉 no goals
              -/


@[congr]
theorem bind_congr' {f g : α → Option β} {x y : Option α} (hx : x = y)
    (hf : ∀ a ∈ y, f a = g a) : x.bind f = y.bind g :=
  hx.symm ▸ bind_congr hf


theorem joinM_eq_join : joinM = @join α :=
  funext fun _ ↦ rfl


theorem bind_eq_bind' {α β : Type u} {f : α → Option β} {x : Option α} : x >>= f = x.bind f :=
  rfl


theorem map_coe {α β} {a : α} {f : α → β} : f <$> (a : Option α) = ↑(f a) :=
  rfl


@[simp]
theorem map_coe' {a : α} {f : α → β} : Option.map f (a : Option α) = ↑(f a) :=
  rfl


/-- `Option.map` as a function between functions is injective. -/
theorem map_injective' : Function.Injective (@Option.map α β) := fun f g h ↦
                                        /-
                                          α : Type u_1
                                          β : Type u_2
                                          f g : α → β
                                          h : Eq (Option.map f) (Option.map g)
                                          x : α
                                          ⊢ Eq (Option.some (f x)) (Option.some (g x))
                                        -/
  funext fun x ↦ some_injective _ <| by simp only [← map_some', h]
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
theorem map_inj {f g : α → β} : Option.map f = Option.map g ↔ f = g :=
  map_injective'.eq_iff


@[simp]
theorem map_eq_id {f : α → α} : Option.map f = id ↔ f = id :=
  map_injective'.eq_iff' map_id


theorem map_comm {f₁ : α → β} {f₂ : α → γ} {g₁ : β → δ} {g₂ : γ → δ} (h : g₁ ∘ f₁ = g₂ ∘ f₂)
    (a : α) :
                                                              /-
                                                                α : Type u_1
                                                                β : Type u_2
                                                                γ : Type u_3
                                                                δ : Type u_4
                                                                f₁ : α → β
                                                                f₂ : α → γ
                                                                g₁ : β → δ
                                                                g₂ : γ → δ
                                                                h : Eq (Function.comp g₁ f₁) (Function.comp g₂ f₂)
                                                                a : α
                                                                ⊢ Eq (Option.map g₁ (Option.map f₁ (Option.some a))) (Option.map g₂ (Option.ma …
                                                              -/
    (Option.map f₁ a).map g₁ = (Option.map f₂ a).map g₂ := by rw [map_map, h, ← map_map]
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem pbind_eq_bind (f : α → Option β) (x : Option α) : (x.pbind fun a _ ↦ f a) = x.bind f := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → Option β
    x : Option α
    ⊢ Eq (x.pbind fun a x => f a) (x.bind f)
  -/
              /-
                🎉 no goals
              -/
  cases x <;> simp only [pbind, none_bind', some_bind']
              /-
                🎉 no goals
              -/


theorem map_bind' (f : β → γ) (x : Option α) (g : α → Option β) :
                                                                      /-
                                                                        α : Type u_1
                                                                        β : Type u_2
                                                                        γ : Type u_3
                                                                        f : β → γ
                                                                        x : Option α
                                                                        g : α → Option β
                                                                        ⊢ Eq (Option.map f (x.bind g)) (x.bind fun a => Option.map f (g a))
                                                                      -/
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
    Option.map f (x.bind g) = x.bind fun a ↦ Option.map f (g a) := by cases x <;> simp
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


theorem pbind_map (f : α → β) (x : Option α) (g : ∀ b : β, b ∈ x.map f → Option γ) :
                                                                                    /-
                                                                                      α : Type u_1
                                                                                      β : Type u_2
                                                                                      γ : Type u_3
                                                                                      f : α → β
                                                                                      x : Option α
                                                                                      g : (b : β) → Membership.mem (Option.map f x) b → Option γ
                                                                                      ⊢ Eq ((Option.map f x).pbind g) (x.pbind fun a h => g (f a) ⋯)
                                                                                    -/
                                                                                                /-
                                                                                                  🎉 no goals
                                                                                                -/
    pbind (Option.map f x) g = x.pbind fun a h ↦ g (f a) (mem_map_of_mem _ h) := by cases x <;> rfl
                                                                                                /-
                                                                                                  🎉 no goals
                                                                                                -/


theorem mem_pmem {a : α} (h : ∀ a ∈ x, p a) (ha : a ∈ x) : f a (h a ha) ∈ pmap f x h := by
  /-
    α : Type u_1
    β : Type u_2
    p : α → Prop
    f : (a : α) → p a → β
    x : Option α
    a : α
    h : ∀ (a : α), Membership.mem x a → p a
    ha : Membership.mem x a
    ⊢ Membership.mem (Option.pmap f x h) (f a ⋯)
  -/
  rw [mem_def] at ha ⊢
  /-
    α : Type u_1
    β : Type u_2
    p : α → Prop
    f : (a : α) → p a → β
    x : Option α
    a : α
    h : ∀ (a : α), Membership.mem x a → p a
    ha✝ : Membership.mem x a
    ha : Eq x (Option.some a)
    ⊢ Eq (Option.pmap f x h) (Option.some (f a ⋯))
  -/
  subst ha
  /-
    α : Type u_1
    β : Type u_2
    p : α → Prop
    f : (a : α) → p a → β
    a : α
    h : ∀ (a_1 : α), Membership.mem (Option.some a) a_1 → p a_1
    ha : Membership.mem (Option.some a) a
    ⊢ Eq (Option.pmap f (Option.some a) h) (Option.some (f a ⋯))
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem pmap_map (g : γ → α) (x : Option γ) (H) :
    pmap f (x.map g) H = pmap (fun a h ↦ f (g a) h) x fun _ h ↦ H _ (mem_map_of_mem _ h) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    p : α → Prop
    f : (a : α) → p a → β
    g : γ → α
    x : Option γ
    H : ∀ (a : α), Membership.mem (Option.map g x) a → p a
    ⊢ Eq (Option.pmap f (Option.map g x) H) (Option.pmap (fun a h => f (g a) h) x ⋯)
  -/
              /-
                🎉 no goals
              -/
  cases x <;> simp only [map_none', map_some', pmap]
              /-
                🎉 no goals
              -/


theorem map_pmap (g : β → γ) (f : ∀ a, p a → β) (x H) :
    Option.map g (pmap f x H) = pmap (fun a h ↦ g (f a h)) x H := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    p : α → Prop
    g : β → γ
    f : (a : α) → p a → β
    x : Option α
    H : ∀ (a : α), Membership.mem x a → p a
    ⊢ Eq (Option.map g (Option.pmap f x H)) (Option.pmap (fun a h => g (f a h)) x H)
  -/
              /-
                🎉 no goals
              -/
  cases x <;> simp only [map_none', map_some', pmap]
              /-
                🎉 no goals
              -/

-- Porting note: Can't simp tag this anymore because `pmap` simplifies
-- @[simp]

theorem pmap_eq_map (p : α → Prop) (f : α → β) (x H) :
    @pmap _ _ p (fun a _ ↦ f a) x H = Option.map f x := by
  /-
    α : Type u_1
    β : Type u_2
    p : α → Prop
    f : α → β
    x : Option α
    H : ∀ (a : α), Membership.mem x a → p a
    ⊢ Eq (Option.pmap (fun a x => f a) x H) (Option.map f x)
  -/
              /-
                🎉 no goals
              -/
  cases x <;> simp only [map_none', map_some', pmap]
              /-
                🎉 no goals
              -/


theorem pmap_bind {α β γ} {x : Option α} {g : α → Option β} {p : β → Prop} {f : ∀ b, p b → γ} (H)
    (H' : ∀ (a : α), ∀ b ∈ g a, b ∈ x >>= g) :
    pmap f (x >>= g) H = x >>= fun a ↦ pmap f (g a) fun _ h ↦ H _ (H' a _ h) := by
  /-
    α β γ : Type u_5
    x : Option α
    g : α → Option β
    p : β → Prop
    f : (b : β) → p b → γ
    H : ∀ (a : β), Membership.mem (Bind.bind x g) a → p a
    H' : ∀ (a : α) (b : β), Membership.mem (g a) b → Membership.mem (Bind.bind x g …
    ⊢ Eq (Option.pmap f (Bind.bind x g) H) (Bind.bind x fun a => Option.pmap f (g  …
  -/
              /-
                🎉 no goals
              -/
  cases x <;> simp only [pmap, bind_eq_bind, none_bind, some_bind]
              /-
                🎉 no goals
              -/


theorem bind_pmap {α β γ} {p : α → Prop} (f : ∀ a, p a → β) (x : Option α) (g : β → Option γ) (H) :
    pmap f x H >>= g = x.pbind fun a h ↦ g (f a (H _ h)) := by
  /-
    α : Type u_5
    β γ : Type u_6
    p : α → Prop
    f : (a : α) → p a → β
    x : Option α
    g : β → Option γ
    H : ∀ (a : α), Membership.mem x a → p a
    ⊢ Eq (Bind.bind (Option.pmap f x H) g) (x.pbind fun a h => g (f a ⋯))
  -/
              /-
                🎉 no goals
              -/
  cases x <;> simp only [pmap, bind_eq_bind, none_bind, some_bind, pbind]
              /-
                🎉 no goals
              -/


theorem pbind_eq_none {f : ∀ a : α, a ∈ x → Option β}
    (h' : ∀ a (H : a ∈ x), f a H = none → x = none) : x.pbind f = none ↔ x = none := by
  /-
    α : Type u_1
    β : Type u_2
    x : Option α
    f : (a : α) → Membership.mem x a → Option β
    h' : ∀ (a : α) (H : Membership.mem x a), Eq (f a H) Option.none → Eq x Option. …
    ⊢ Iff (Eq (x.pbind f) Option.none) (Eq x Option.none)
  -/
  cases x
    /-
      case none
      α : Type u_1
      β : Type u_2
      f : (a : α) → Membership.mem Option.none a → Option β
      h' : ∀ (a : α) (H : Membership.mem Option.none a), Eq (f a H) Option.none → Eq …
      ⊢ Iff (Eq (Option.none.pbind f) Option.none) (Eq Option.none Option.none)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case some
      α : Type u_1
      β : Type u_2
      val✝ : α
      f : (a : α) → Membership.mem (Option.some val✝) a → Option β
      h' : ∀ (a : α) (H : Membership.mem (Option.some val✝) a), Eq (f a H) Option.no …
      ⊢ Iff (Eq ((Option.some val✝).pbind f) Option.none) (Eq (Option.some val✝) Opt …
    -/
  · simp only [pbind, iff_false, reduceCtorEq]
    /-
      case some
      α : Type u_1
      β : Type u_2
      val✝ : α
      f : (a : α) → Membership.mem (Option.some val✝) a → Option β
      h' : ∀ (a : α) (H : Membership.mem (Option.some val✝) a), Eq (f a H) Option.no …
      ⊢ Not (Eq (f val✝ ⋯) Option.none)
    -/
    intro h
    /-
      case some
      α : Type u_1
      β : Type u_2
      val✝ : α
      f : (a : α) → Membership.mem (Option.some val✝) a → Option β
      h' : ∀ (a : α) (H : Membership.mem (Option.some val✝) a), Eq (f a H) Option.no …
      h : Eq (f val✝ ⋯) Option.none
      ⊢ False
    -/
    cases h' _ rfl h
    /-
      🎉 no goals
    -/


theorem pbind_eq_some {f : ∀ a : α, a ∈ x → Option β} {y : β} :
    x.pbind f = some y ↔ ∃ (z : α) (H : z ∈ x), f z H = some y := by
  /-
    α : Type u_1
    β : Type u_2
    x : Option α
    f : (a : α) → Membership.mem x a → Option β
    y : β
    ⊢ Iff (Eq (x.pbind f) (Option.some y)) (Exists fun z => Exists fun H => Eq (f  …
  -/
  rcases x with (_|x)
    /-
      case none
      α : Type u_1
      β : Type u_2
      y : β
      f : (a : α) → Membership.mem Option.none a → Option β
      ⊢ Iff (Eq (Option.none.pbind f) (Option.some y)) (Exists fun z => Exists fun H …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case some
      α : Type u_1
      β : Type u_2
      y : β
      x : α
      f : (a : α) → Membership.mem (Option.some x) a → Option β
      ⊢ Iff (Eq ((Option.some x).pbind f) (Option.some y)) (Exists fun z => Exists f …
    -/
  · simp only [pbind]
    /-
      case some
      α : Type u_1
      β : Type u_2
      y : β
      x : α
      f : (a : α) → Membership.mem (Option.some x) a → Option β
      ⊢ Iff (Eq (f x ⋯) (Option.some y)) (Exists fun z => Exists fun H => Eq (f z H) …
    -/
    refine ⟨fun h ↦ ⟨x, rfl, h⟩, ?_⟩
    /-
      case some
      α : Type u_1
      β : Type u_2
      y : β
      x : α
      f : (a : α) → Membership.mem (Option.some x) a → Option β
      ⊢ (Exists fun z => Exists fun H => Eq (f z H) (Option.some y)) → Eq (f x ⋯) (O …
    -/
    rintro ⟨z, H, hz⟩
    /-
      case some.intro.intro
      α : Type u_1
      β : Type u_2
      y : β
      x : α
      f : (a : α) → Membership.mem (Option.some x) a → Option β
      z : α
      H : Membership.mem (Option.some x) z
      hz : Eq (f z H) (Option.some y)
      ⊢ Eq (f x ⋯) (Option.some y)
    -/
    simp only [mem_def, Option.some_inj] at H
    /-
      case some.intro.intro
      α : Type u_1
      β : Type u_2
      y : β
      x : α
      f : (a : α) → Membership.mem (Option.some x) a → Option β
      z : α
      H✝ : Membership.mem (Option.some x) z
      hz : Eq (f z H✝) (Option.some y)
      H : Eq x z
      ⊢ Eq (f x ⋯) (Option.some y)
    -/
    simpa [H] using hz
    /-
      🎉 no goals
    -/

-- Porting note: Can't simp tag this anymore because `join` and `pmap` simplify
-- @[simp]

theorem join_pmap_eq_pmap_join {f : ∀ a, p a → β} {x : Option (Option α)} (H) :
    (pmap (pmap f) x H).join = pmap f x.join fun a h ↦ H (some a) (mem_of_mem_join h) _ rfl := by
  /-
    α : Type u_1
    β : Type u_2
    p : α → Prop
    f : (a : α) → p a → β
    x : Option (Option α)
    H : ∀ (a : Option α), Membership.mem x a → ∀ (a_2 : α), Membership.mem a a_2 → …
    ⊢ Eq (Option.pmap (Option.pmap f) x H).join (Option.pmap f x.join ⋯)
  -/
                                /-
                                  🎉 no goals
                                -/
                                /-
                                  🎉 no goals
                                -/
  rcases x with (_ | _ | x) <;> simp
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem seq_some {α β} {a : α} {f : α → β} : some f <*> some a = some (f a) :=
  rfl


@[simp]
theorem some_orElse' (a : α) (x : Option α) : (some a).orElse (fun _ ↦ x) = some a :=
  rfl


@[simp]
                                                                        /-
                                                                          α : Type u_1
                                                                          x : Option α
                                                                          ⊢ Eq (Option.none.orElse fun x_1 => x) x
                                                                        -/
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
theorem none_orElse' (x : Option α) : none.orElse (fun _ ↦ x) = x := by cases x <;> rfl
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


@[simp]
                                                                        /-
                                                                          α : Type u_1
                                                                          x : Option α
                                                                          ⊢ Eq (x.orElse fun x => Option.none) x
                                                                        -/
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
theorem orElse_none' (x : Option α) : x.orElse (fun _ ↦ none) = x := by cases x <;> rfl
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


theorem exists_ne_none {p : Option α → Prop} : (∃ x ≠ none, p x) ↔ (∃ x : α, p x) := by
  /-
    α : Type u_1
    p : Option α → Prop
    ⊢ Iff (Exists fun x => And (Ne x Option.none) (p x)) (Exists fun x => p (Optio …
  -/
  simp only [← exists_prop, bex_ne_none]
  /-
    🎉 no goals
  -/


theorem iget_mem [Inhabited α] : ∀ {o : Option α}, isSome o → o.iget ∈ o
  | some _, _ => rfl


theorem iget_of_mem [Inhabited α] {a : α} : ∀ {o : Option α}, a ∈ o → o.iget = a
  | _, rfl => rfl


theorem getD_default_eq_iget [Inhabited α] (o : Option α) :
                                  /-
                                    α : Type u_1
                                    inst✝ : Inhabited α
                                    o : Option α
                                    ⊢ Eq (o.getD Inhabited.default) o.iget
                                  -/
                                              /-
                                                🎉 no goals
                                              -/
    o.getD default = o.iget := by cases o <;> rfl
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem guard_eq_some' {p : Prop} [Decidable p] (u) : _root_.guard p = some u ↔ p := by
  /-
    p : Prop
    inst✝ : Decidable p
    u : Unit
    ⊢ Iff (Eq (_root_.guard p) (Option.some u)) p
  -/
  cases u
  /-
    case unit
    p : Prop
    inst✝ : Decidable p
    ⊢ Iff (Eq (_root_.guard p) (Option.some PUnit.unit)) p
  -/
                     /-
                       🎉 no goals
                     -/
  by_cases h : p <;> simp [_root_.guard, h]
                     /-
                       🎉 no goals
                     -/


theorem liftOrGet_choice {f : α → α → α} (h : ∀ a b, f a b = a ∨ f a b = b) :
    ∀ o₁ o₂, liftOrGet f o₁ o₂ = o₁ ∨ liftOrGet f o₁ o₂ = o₂
  | none, none => Or.inl rfl
  | some _, none => Or.inl rfl
  | none, some _ => Or.inr rfl
                         /-
                           α : Type u_1
                           f : α → α → α
                           h : ∀ (a b : α), Or (Eq (f a b) a) (Eq (f a b) b)
                           a b : α
                           ⊢ Or (Eq (Option.liftOrGet f (Option.some a) (Option.some b)) (Option.some a)) …
                         -/
  | some a, some b => by simpa [liftOrGet] using h a b
                         /-
                           🎉 no goals
                         -/


/-- Given an element of `a : Option α`, a default element `b : β` and a function `α → β`, apply this
function to `a` if it comes from `α`, and return `b` otherwise. -/
def casesOn' : Option α → β → (α → β) → β
  | none, n, _ => n
  | some a, _, s => s a


@[simp]
theorem casesOn'_none (x : β) (f : α → β) : casesOn' none x f = x :=
  rfl


@[simp]
theorem casesOn'_some (x : β) (f : α → β) (a : α) : casesOn' (some a) x f = f a :=
  rfl


@[simp]
theorem casesOn'_coe (x : β) (f : α → β) (a : α) : casesOn' (a : Option α) x f = f a :=
  rfl

-- Porting note: Left-hand side does not simplify.
-- @[simp]

theorem casesOn'_none_coe (f : Option α → β) (o : Option α) :
                                                       /-
                                                         α : Type u_1
                                                         β : Type u_2
                                                         f : Option α → β
                                                         o : Option α
                                                         ⊢ Eq (o.casesOn' (f Option.none) (Function.comp f fun a => Option.some a)) (f o)
                                                       -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
    casesOn' o (f none) (f ∘ (fun a ↦ ↑a)) = f o := by cases o <;> rfl
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


lemma casesOn'_eq_elim (b : β) (f : α → β) (a : Option α) :
                                                    /-
                                                      α : Type u_1
                                                      β : Type u_2
                                                      b : β
                                                      f : α → β
                                                      a : Option α
                                                      ⊢ Eq (a.casesOn' b f) (a.elim b f)
                                                    -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
    Option.casesOn' a b f = Option.elim a b f := by cases a <;> rfl
                                                                /-
                                                                  🎉 no goals
                                                                -/

-- porting note: workaround for https://github.com/leanprover/lean4/issues/2049

compile_inductive% Option


theorem orElse_eq_some (o o' : Option α) (x : α) :
    (o <|> o') = some x ↔ o = some x ∨ o = none ∧ o' = some x := by
  /-
    α : Type u_1
    o o' : Option α
    x : α
    ⊢ Iff (Eq (HOrElse.hOrElse o fun x => o') (Option.some x)) (Or (Eq o (Option.s …
  -/
  cases o
    /-
      case none
      α : Type u_1
      o' : Option α
      x : α
      ⊢ Iff (Eq (HOrElse.hOrElse Option.none fun x => o') (Option.some x)) (Or (Eq O …
    -/
  · simp only [true_and, false_or, eq_self_iff_true, none_orElse, reduceCtorEq]
    /-
      🎉 no goals
    -/
    /-
      case some
      α : Type u_1
      o' : Option α
      x val✝ : α
      ⊢ Iff (Eq (HOrElse.hOrElse (Option.some val✝) fun x => o') (Option.some x)) (O …
    -/
  · simp only [some_orElse, or_false, false_and, reduceCtorEq]
    /-
      🎉 no goals
    -/



theorem orElse_eq_some' (o o' : Option α) (x : α) :
    o.orElse (fun _ ↦ o') = some x ↔ o = some x ∨ o = none ∧ o' = some x :=
  Option.orElse_eq_some o o' x


@[simp]
theorem orElse_eq_none (o o' : Option α) : (o <|> o') = none ↔ o = none ∧ o' = none := by
  /-
    α : Type u_1
    o o' : Option α
    ⊢ Iff (Eq (HOrElse.hOrElse o fun x => o') Option.none) (And (Eq o Option.none) …
  -/
  cases o
    /-
      case none
      α : Type u_1
      o' : Option α
      ⊢ Iff (Eq (HOrElse.hOrElse Option.none fun x => o') Option.none) (And (Eq Opti …
    -/
  · simp only [true_and, none_orElse, eq_self_iff_true]
    /-
      🎉 no goals
    -/
    /-
      case some
      α : Type u_1
      o' : Option α
      val✝ : α
      ⊢ Iff (Eq (HOrElse.hOrElse (Option.some val✝) fun x => o') Option.none) (And ( …
    -/
  · simp only [some_orElse, reduceCtorEq, false_and]
    /-
      🎉 no goals
    -/


@[simp]
theorem orElse_eq_none' (o o' : Option α) : o.orElse (fun _ ↦ o') = none ↔ o = none ∧ o' = none :=
  Option.orElse_eq_none o o'


theorem choice_eq_none (α : Type*) [IsEmpty α] : choice α = none :=
  dif_neg (not_nonempty_iff_imp_false.mpr isEmptyElim)


theorem elim_none_some (f : Option α → β) : (fun x ↦ Option.elim x (f none) (f ∘ some)) = f :=
                    /-
                      α : Type u_1
                      β : Type u_2
                      f : Option α → β
                      o : Option α
                      ⊢ Eq (o.elim (f Option.none) (Function.comp f Option.some)) (f o)
                    -/
                                /-
                                  🎉 no goals
                                -/
  funext fun o ↦ by cases o <;> rfl
                                /-
                                  🎉 no goals
                                -/


theorem elim_comp (h : α → β) {f : γ → α} {x : α} {i : Option γ} :
                                                           /-
                                                             α : Type u_1
                                                             β : Type u_2
                                                             γ : Type u_3
                                                             h : α → β
                                                             f : γ → α
                                                             x : α
                                                             i : Option γ
                                                             ⊢ Eq (i.elim (h x) fun j => h (f j)) (h (i.elim x f))
                                                           -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
    (i.elim (h x) fun j => h (f j)) = h (i.elim x f) := by cases i <;> rfl
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem elim_comp₂ (h : α → β → γ) {f : γ → α} {x : α} {g : γ → β} {y : β}
    {i : Option γ} : (i.elim (h x y) fun j => h (f j) (g j)) = h (i.elim x f) (i.elim y g) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    h : α → β → γ
    f : γ → α
    x : α
    g : γ → β
    y : β
    i : Option γ
    ⊢ Eq (i.elim (h x y) fun j => h (f j) (g j)) (h (i.elim x f) (i.elim y g))
  -/
              /-
                🎉 no goals
              -/
  cases i <;> rfl
              /-
                🎉 no goals
              -/


theorem elim_apply {f : γ → α → β} {x : α → β} {i : Option γ} {y : α} :
                                                     /-
                                                       α : Type u_1
                                                       β : Type u_2
                                                       γ : Type u_3
                                                       f : γ → α → β
                                                       x : α → β
                                                       i : Option γ
                                                       y : α
                                                       ⊢ Eq (i.elim x f y) (i.elim (x y) fun j => f j y)
                                                     -/
    i.elim x f y = i.elim (x y) fun j => f j y := by rw [elim_comp fun f : α → β => f y]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
lemma bnot_isSome (a : Option α) : (! a.isSome) = a.isNone := by
  /-
    α : Type u_1
    a : Option α
    ⊢ Eq a.isSome.not a.isNone
  -/
              /-
                🎉 no goals
              -/
  cases a <;> simp
              /-
                🎉 no goals
              -/


@[simp]
lemma bnot_comp_isSome : (! ·) ∘ @Option.isSome α = Option.isNone := by
  /-
    α : Type u_1
    ⊢ Eq (Function.comp (fun x => x.not) Option.isSome) Option.isNone
  -/
  funext
  /-
    case h
    α : Type u_1
    x✝ : Option α
    ⊢ Eq (Function.comp (fun x => x.not) Option.isSome x✝) x✝.isNone
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma bnot_isNone (a : Option α) : (! a.isNone) = a.isSome := by
  /-
    α : Type u_1
    a : Option α
    ⊢ Eq a.isNone.not a.isSome
  -/
              /-
                🎉 no goals
              -/
  cases a <;> simp
              /-
                🎉 no goals
              -/


@[simp]
lemma bnot_comp_isNone : (! ·) ∘ @Option.isNone α = Option.isSome := by
  /-
    α : Type u_1
    ⊢ Eq (Function.comp (fun x => x.not) Option.isNone) Option.isSome
  -/
  funext x
  /-
    case h
    α : Type u_1
    x : Option α
    ⊢ Eq (Function.comp (fun x => x.not) Option.isNone x) x.isSome
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma isNone_eq_false_iff (a : Option α) : Option.isNone a = false ↔ Option.isSome a := by
  /-
    α : Type u_1
    a : Option α
    ⊢ Iff (Eq a.isNone Bool.false) (Eq a.isSome Bool.true)
  -/
              /-
                🎉 no goals
              -/
  cases a <;> simp
              /-
                🎉 no goals
              -/


lemma eq_none_or_eq_some (a : Option α) : a = none ∨ ∃ x, a = some x :=
  Option.exists.mp exists_eq'


lemma forall_some_ne_iff_eq_none {o : Option α} : (∀ (x : α), some x ≠ o) ↔ o = none := by
  /-
    α : Type u_1
    o : Option α
    ⊢ Iff (∀ (x : α), Ne (Option.some x) o) (Eq o Option.none)
  -/
  apply not_iff_not.1
  /-
    α : Type u_1
    o : Option α
    ⊢ Iff (Not (∀ (x : α), Ne (Option.some x) o)) (Not (Eq o Option.none))
  -/
  simpa only [not_forall, not_not] using Option.ne_none_iff_exists.symm
  /-
    🎉 no goals
  -/


