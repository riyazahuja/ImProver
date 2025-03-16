/-- A language is a set of strings over an alphabet. -/
def Language (α) :=
  Set (List α)


instance : Membership (List α) (Language α) := ⟨Set.Mem⟩

instance : Singleton (List α) (Language α) := ⟨Set.singleton⟩

instance : Insert (List α) (Language α) := ⟨Set.insert⟩

instance instCompleteAtomicBooleanAlgebra : CompleteAtomicBooleanAlgebra (Language α) :=
  Set.instCompleteAtomicBooleanAlgebra


/-- Zero language has no elements. -/
instance : Zero (Language α) :=
  ⟨(∅ : Set _)⟩


/-- `1 : Language α` contains only one element `[]`. -/
instance : One (Language α) :=
  ⟨{[]}⟩


instance : Inhabited (Language α) := ⟨(∅ : Set _)⟩


/-- The sum of two languages is their union. -/
instance : Add (Language α) :=
  ⟨((· ∪ ·) : Set (List α) → Set (List α) → Set (List α))⟩


/-- The product of two languages `l` and `m` is the language made of the strings `x ++ y` where
`x ∈ l` and `y ∈ m`. -/
instance : Mul (Language α) :=
  ⟨image2 (· ++ ·)⟩


theorem zero_def : (0 : Language α) = (∅ : Set _) :=
  rfl


theorem one_def : (1 : Language α) = ({[]} : Set (List α)) :=
  rfl


theorem add_def (l m : Language α) : l + m = (l ∪ m : Set (List α)) :=
  rfl


theorem mul_def (l m : Language α) : l * m = image2 (· ++ ·) l m :=
  rfl


/-- The Kleene star of a language `L` is the set of all strings which can be written by
concatenating strings from `L`. -/
instance : KStar (Language α) := ⟨fun l ↦ {x | ∃ L : List (List α), x = L.flatten ∧ ∀ y ∈ L, y ∈ l}⟩


lemma kstar_def (l : Language α) : l∗ = {x | ∃ L : List (List α), x = L.flatten ∧ ∀ y ∈ L, y ∈ l} :=
  rfl

-- Porting note: `reducible` attribute cannot be local,
--               so this new theorem is required in place of `Set.ext`.

@[ext]
theorem ext {l m : Language α} (h : ∀ (x : List α), x ∈ l ↔ x ∈ m) : l = m :=
  Set.ext h


@[simp]
theorem not_mem_zero (x : List α) : x ∉ (0 : Language α) :=
  id


@[simp]
                                                                   /-
                                                                     α : Type u_1
                                                                     x : List α
                                                                     ⊢ Iff (Membership.mem 1 x) (Eq x List.nil)
                                                                   -/
theorem mem_one (x : List α) : x ∈ (1 : Language α) ↔ x = [] := by rfl
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


theorem nil_mem_one : [] ∈ (1 : Language α) :=
  Set.mem_singleton _


theorem mem_add (l m : Language α) (x : List α) : x ∈ l + m ↔ x ∈ l ∨ x ∈ m :=
  Iff.rfl


theorem mem_mul : x ∈ l * m ↔ ∃ a ∈ l, ∃ b ∈ m, a ++ b = x :=
  mem_image2


theorem append_mem_mul : a ∈ l → b ∈ m → a ++ b ∈ l * m :=
  mem_image2_of_mem


theorem mem_kstar : x ∈ l∗ ↔ ∃ L : List (List α), x = L.flatten ∧ ∀ y ∈ L, y ∈ l :=
  Iff.rfl


theorem join_mem_kstar {L : List (List α)} (h : ∀ y ∈ L, y ∈ l) : L.flatten ∈ l∗ :=
  ⟨L, rfl, h⟩


theorem nil_mem_kstar (l : Language α) : [] ∈ l∗ :=
                         /-
                           α : Type u_1
                           l : Language α
                           x✝ : List α
                           h : Membership.mem List.nil x✝
                           ⊢ Membership.mem l x✝
                         -/
  ⟨[], rfl, fun _ h ↦ by contradiction⟩
                         /-
                           🎉 no goals
                         -/


instance instSemiring : Semiring (Language α) where
  add := (· + ·)
  add_assoc := union_assoc
  zero := 0
  zero_add := empty_union
  add_zero := union_empty
  add_comm := union_comm
  mul := (· * ·)
  mul_assoc _ _ _ := image2_assoc append_assoc
  zero_mul _ := image2_empty_left
  mul_zero _ := image2_empty_right
  one := 1
                  /-
                    α : Type u_1
                    β : Type u_2
                    γ : Type u_3
                    l✝ m : Language α
                    a b x : List α
                    l : Language α
                    ⊢ Eq (HMul.hMul 1 l) l
                  -/
  one_mul l := by simp [mul_def, one_def]
                  /-
                    🎉 no goals
                  -/
                  /-
                    α : Type u_1
                    β : Type u_2
                    γ : Type u_3
                    l✝ m : Language α
                    a b x : List α
                    l : Language α
                    ⊢ Eq (HMul.hMul l 1) l
                  -/
  mul_one l := by simp [mul_def, one_def]
                  /-
                    🎉 no goals
                  -/
  natCast n := if n = 0 then 0 else 1
  natCast_zero := rfl
                       /-
                         α : Type u_1
                         β : Type u_2
                         γ : Type u_3
                         l m : Language α
                         a b x : List α
                         n : Nat
                         ⊢ Eq (NatCast.natCast (HAdd.hAdd n 1)) (HAdd.hAdd (NatCast.natCast n) 1)
                       -/
                                   /-
                                     🎉 no goals
                                   -/
  natCast_succ n := by cases n <;> simp [Nat.cast, add_def, zero_def]
                                   /-
                                     🎉 no goals
                                   -/
  left_distrib _ _ _ := image2_union_right
  right_distrib _ _ _ := image2_union_left
  nsmul := nsmulRec


@[simp]
theorem add_self (l : Language α) : l + l = l :=
  sup_idem _


/-- Maps the alphabet of a language. -/
def map (f : α → β) : Language α →+* Language β where
  toFun := image (List.map f)
  map_zero' := image_empty _
  map_one' := image_singleton
  map_add' := image_union _
  map_mul' _ _ := image_image2_distrib <| map_append _


@[simp]
                                                     /-
                                                       α : Type u_1
                                                       l : Language α
                                                       ⊢ Eq ((Language.map id) l) l
                                                     -/
theorem map_id (l : Language α) : map id l = l := by simp [map]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
theorem map_map (g : β → γ) (f : α → β) (l : Language α) : map g (map f l) = map (g ∘ f) l := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    g : β → γ
    f : α → β
    l : Language α
    ⊢ Eq ((Language.map g) ((Language.map f) l)) ((Language.map (Function.comp g f …
  -/
  simp [map, image_image]
  /-
    🎉 no goals
  -/


lemma mem_kstar_iff_exists_nonempty {x : List α} :
    x ∈ l∗ ↔ ∃ S : List (List α), x = S.flatten ∧ ∀ y ∈ S, y ∈ l ∧ y ≠ [] := by
  /-
    α : Type u_1
    l : Language α
    x : List α
    ⊢ Iff (Membership.mem (KStar.kstar l) x) (Exists fun S => And (Eq x S.flatten) …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      l : Language α
      x : List α
      ⊢ Membership.mem (KStar.kstar l) x → Exists fun S => And (Eq x S.flatten) (∀ ( …
    -/
  · rintro ⟨S, rfl, h⟩
    refine ⟨S.filter fun l ↦ !List.isEmpty l,
      by simp [List.flatten_filter_not_isEmpty], fun y hy ↦ ?_⟩
    -- Porting note: The previous code was:
    -- rw [mem_filter, empty_iff_eq_nil] at hy
    /-
      case mp.intro.intro
      α : Type u_1
      l : Language α
      S : List (List α)
      h : ∀ (y : List α), Membership.mem S y → Membership.mem l y
      y : List α
      hy : Membership.mem (List.filter (fun l => l.isEmpty.not) S) y
      ⊢ And (Membership.mem l y) (Ne y List.nil)
    -/
    rw [mem_filter, Bool.not_eq_true', ← Bool.bool_iff_false, List.isEmpty_iff] at hy
    /-
      case mp.intro.intro
      α : Type u_1
      l : Language α
      S : List (List α)
      h : ∀ (y : List α), Membership.mem S y → Membership.mem l y
      y : List α
      hy : And (Membership.mem S y) (Not (Eq y List.nil))
      ⊢ And (Membership.mem l y) (Ne y List.nil)
    -/
    exact ⟨h y hy.1, hy.2⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      l : Language α
      x : List α
      ⊢ (Exists fun S => And (Eq x S.flatten) (∀ (y : List α), Membership.mem S y →  …
    -/
  · rintro ⟨S, hx, h⟩
    /-
      case mpr.intro.intro
      α : Type u_1
      l : Language α
      x : List α
      S : List (List α)
      hx : Eq x S.flatten
      h : ∀ (y : List α), Membership.mem S y → And (Membership.mem l y) (Ne y List.n …
      ⊢ Membership.mem (KStar.kstar l) x
    -/
    exact ⟨S, hx, fun y hy ↦ (h y hy).1⟩
    /-
      🎉 no goals
    -/


theorem kstar_def_nonempty (l : Language α) :
    l∗ = { x | ∃ S : List (List α), x = S.flatten ∧ ∀ y ∈ S, y ∈ l ∧ y ≠ [] } := by
  /-
    α : Type u_1
    l : Language α
    ⊢ Eq (KStar.kstar l) (setOf fun x => Exists fun S => And (Eq x S.flatten) (∀ ( …
  -/
  ext x; apply mem_kstar_iff_exists_nonempty
         /-
           🎉 no goals
         -/


theorem le_iff (l m : Language α) : l ≤ m ↔ l + m = m :=
  sup_eq_right.symm


theorem le_mul_congr {l₁ l₂ m₁ m₂ : Language α} : l₁ ≤ m₁ → l₂ ≤ m₂ → l₁ * l₂ ≤ m₁ * m₂ := by
  /-
    α : Type u_1
    l₁ l₂ m₁ m₂ : Language α
    ⊢ LE.le l₁ m₁ → LE.le l₂ m₂ → LE.le (HMul.hMul l₁ l₂) (HMul.hMul m₁ m₂)
  -/
  intro h₁ h₂ x hx
  /-
    α : Type u_1
    l₁ l₂ m₁ m₂ : Language α
    h₁ : LE.le l₁ m₁
    h₂ : LE.le l₂ m₂
    x : List α
    hx : Membership.mem (HMul.hMul l₁ l₂) x
    ⊢ Membership.mem (HMul.hMul m₁ m₂) x
  -/
  simp only [mul_def, exists_and_left, mem_image2, image_prod] at hx ⊢
  /-
    α : Type u_1
    l₁ l₂ m₁ m₂ : Language α
    h₁ : LE.le l₁ m₁
    h₂ : LE.le l₂ m₂
    x : List α
    hx : Exists fun a => And (Membership.mem l₁ a) (Exists fun b => And (Membershi …
    ⊢ Exists fun a => And (Membership.mem m₁ a) (Exists fun b => And (Membership.m …
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem le_add_congr {l₁ l₂ m₁ m₂ : Language α} : l₁ ≤ m₁ → l₂ ≤ m₂ → l₁ + l₂ ≤ m₁ + m₂ :=
  sup_le_sup


theorem mem_iSup {ι : Sort v} {l : ι → Language α} {x : List α} : (x ∈ ⨆ i, l i) ↔ ∃ i, x ∈ l i :=
  mem_iUnion


theorem iSup_mul {ι : Sort v} (l : ι → Language α) (m : Language α) :
    (⨆ i, l i) * m = ⨆ i, l i * m :=
  image2_iUnion_left _ _ _


theorem mul_iSup {ι : Sort v} (l : ι → Language α) (m : Language α) :
    (m * ⨆ i, l i) = ⨆ i, m * l i :=
  image2_iUnion_right _ _ _


theorem iSup_add {ι : Sort v} [Nonempty ι] (l : ι → Language α) (m : Language α) :
    (⨆ i, l i) + m = ⨆ i, l i + m :=
  iSup_sup


theorem add_iSup {ι : Sort v} [Nonempty ι] (l : ι → Language α) (m : Language α) :
    (m + ⨆ i, l i) = ⨆ i, m + l i :=
  sup_iSup


theorem mem_pow {l : Language α} {x : List α} {n : ℕ} :
    x ∈ l ^ n ↔ ∃ S : List (List α), x = S.flatten ∧ S.length = n ∧ ∀ y ∈ S, y ∈ l := by
  /-
    α : Type u_1
    l : Language α
    x : List α
    n : Nat
    ⊢ Iff (Membership.mem (HPow.hPow l n) x) (Exists fun S => And (Eq x S.flatten) …
  -/
  induction' n with n ihn generalizing x
    /-
      case zero
      α : Type u_1
      l : Language α
      x : List α
      ⊢ Iff (Membership.mem (HPow.hPow l 0) x) (Exists fun S => And (Eq x S.flatten) …
    -/
  · simp only [mem_one, pow_zero, length_eq_zero]
    /-
      case zero
      α : Type u_1
      l : Language α
      x : List α
      ⊢ Iff (Eq x List.nil) (Exists fun S => And (Eq x S.flatten) (And (Eq S List.ni …
    -/
    constructor
      /-
        case zero.mp
        α : Type u_1
        l : Language α
        x : List α
        ⊢ Eq x List.nil → Exists fun S => And (Eq x S.flatten) (And (Eq S List.nil) (∀ …
      -/
    · rintro rfl
      /-
        case zero.mp
        α : Type u_1
        l : Language α
        ⊢ Exists fun S => And (Eq List.nil S.flatten) (And (Eq S List.nil) (∀ (y : Lis …
      -/
      exact ⟨[], rfl, rfl, fun _ h ↦ by contradiction⟩
      /-
        🎉 no goals
      -/
      /-
        case zero.mpr
        α : Type u_1
        l : Language α
        x : List α
        ⊢ (Exists fun S => And (Eq x S.flatten) (And (Eq S List.nil) (∀ (y : List α),  …
      -/
    · rintro ⟨_, rfl, rfl, _⟩
      /-
        case zero.mpr.intro.intro.intro
        α : Type u_1
        l : Language α
        right✝ : ∀ (y : List α), Membership.mem List.nil y → Membership.mem l y
        ⊢ Eq List.nil.flatten List.nil
      -/
      rfl
      /-
        🎉 no goals
      -/
    /-
      case succ
      α : Type u_1
      l : Language α
      n : Nat
      ihn : ∀ {x : List α}, Iff (Membership.mem (HPow.hPow l n) x) (Exists fun S =>  …
      x : List α
      ⊢ Iff (Membership.mem (HPow.hPow l (HAdd.hAdd n 1)) x) (Exists fun S => And (E …
    -/
  · simp only [pow_succ', mem_mul, ihn]
    /-
      case succ
      α : Type u_1
      l : Language α
      n : Nat
      ihn : ∀ {x : List α}, Iff (Membership.mem (HPow.hPow l n) x) (Exists fun S =>  …
      x : List α
      ⊢ Iff (Exists fun a => And (Membership.mem l a) (Exists fun b => And (Exists f …
    -/
    constructor
      /-
        case succ.mp
        α : Type u_1
        l : Language α
        n : Nat
        ihn : ∀ {x : List α}, Iff (Membership.mem (HPow.hPow l n) x) (Exists fun S =>  …
        x : List α
        ⊢ (Exists fun a => And (Membership.mem l a) (Exists fun b => And (Exists fun S …
      -/
    · rintro ⟨a, ha, b, ⟨S, rfl, rfl, hS⟩, rfl⟩
      /-
        case succ.mp.intro.intro.intro.intro.intro.intro.intro
        α : Type u_1
        l : Language α
        a : List α
        ha : Membership.mem l a
        S : List (List α)
        hS : ∀ (y : List α), Membership.mem S y → Membership.mem l y
        ihn : ∀ {x : List α}, Iff (Membership.mem (HPow.hPow l S.length) x) (Exists fu …
        ⊢ Exists fun S_1 => And (Eq (HAppend.hAppend a S.flatten) S_1.flatten) (And (E …
      -/
      exact ⟨a :: S, rfl, rfl, forall_mem_cons.2 ⟨ha, hS⟩⟩
      /-
        🎉 no goals
      -/
      /-
        case succ.mpr
        α : Type u_1
        l : Language α
        n : Nat
        ihn : ∀ {x : List α}, Iff (Membership.mem (HPow.hPow l n) x) (Exists fun S =>  …
        x : List α
        ⊢ (Exists fun S => And (Eq x S.flatten) (And (Eq S.length (HAdd.hAdd n 1)) (∀  …
      -/
                                           /-
                                             🎉 no goals
                                           -/
    · rintro ⟨_ | ⟨a, S⟩, rfl, hn, hS⟩ <;> cases hn
      /-
        case succ.mpr.intro.cons.intro.intro.refl
        α : Type u_1
        l : Language α
        a : List α
        S : List (List α)
        hS : ∀ (y : List α), Membership.mem (List.cons a S) y → Membership.mem l y
        ihn : ∀ {x : List α}, Iff (Membership.mem (HPow.hPow l S.length) x) (Exists fu …
        ⊢ Exists fun a_1 => And (Membership.mem l a_1) (Exists fun b => And (Exists fu …
      -/
      rw [forall_mem_cons] at hS
      /-
        case succ.mpr.intro.cons.intro.intro.refl
        α : Type u_1
        l : Language α
        a : List α
        S : List (List α)
        hS : And (Membership.mem l a) (∀ (x : List α), Membership.mem S x → Membership …
        ihn : ∀ {x : List α}, Iff (Membership.mem (HPow.hPow l S.length) x) (Exists fu …
        ⊢ Exists fun a_1 => And (Membership.mem l a_1) (Exists fun b => And (Exists fu …
      -/
      exact ⟨a, hS.1, _, ⟨S, rfl, rfl, hS.2⟩, rfl⟩
      /-
        🎉 no goals
      -/


theorem kstar_eq_iSup_pow (l : Language α) : l∗ = ⨆ i : ℕ, l ^ i := by
  /-
    α : Type u_1
    l : Language α
    ⊢ Eq (KStar.kstar l) (iSup fun i => HPow.hPow l i)
  -/
  ext x
  /-
    case h
    α : Type u_1
    l : Language α
    x : List α
    ⊢ Iff (Membership.mem (KStar.kstar l) x) (Membership.mem (iSup fun i => HPow.h …
  -/
  simp only [mem_kstar, mem_iSup, mem_pow]
  /-
    case h
    α : Type u_1
    l : Language α
    x : List α
    ⊢ Iff (Exists fun L => And (Eq x L.flatten) (∀ (y : List α), Membership.mem L  …
  -/
  constructor
    /-
      case h.mp
      α : Type u_1
      l : Language α
      x : List α
      ⊢ (Exists fun L => And (Eq x L.flatten) (∀ (y : List α), Membership.mem L y →  …
    -/
  · rintro ⟨S, rfl, hS⟩
    /-
      case h.mp.intro.intro
      α : Type u_1
      l : Language α
      S : List (List α)
      hS : ∀ (y : List α), Membership.mem S y → Membership.mem l y
      ⊢ Exists fun i => Exists fun S_1 => And (Eq S.flatten S_1.flatten) (And (Eq S_ …
    -/
    exact ⟨_, S, rfl, rfl, hS⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      α : Type u_1
      l : Language α
      x : List α
      ⊢ (Exists fun i => Exists fun S => And (Eq x S.flatten) (And (Eq S.length i) ( …
    -/
  · rintro ⟨_, S, rfl, rfl, hS⟩
    /-
      case h.mpr.intro.intro.intro.intro
      α : Type u_1
      l : Language α
      S : List (List α)
      hS : ∀ (y : List α), Membership.mem S y → Membership.mem l y
      ⊢ Exists fun L => And (Eq S.flatten L.flatten) (∀ (y : List α), Membership.mem …
    -/
    exact ⟨S, rfl, hS⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem map_kstar (f : α → β) (l : Language α) : map f l∗ = (map f l)∗ := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    l : Language α
    ⊢ Eq ((Language.map f) (KStar.kstar l)) (KStar.kstar ((Language.map f) l))
  -/
  rw [kstar_eq_iSup_pow, kstar_eq_iSup_pow]
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    l : Language α
    ⊢ Eq ((Language.map f) (iSup fun i => HPow.hPow l i)) (iSup fun i => HPow.hPow …
  -/
  simp_rw [← map_pow]
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    l : Language α
    ⊢ Eq ((Language.map f) (iSup fun i => HPow.hPow l i)) (iSup fun i => (Language …
  -/
  exact image_iUnion
  /-
    🎉 no goals
  -/


theorem mul_self_kstar_comm (l : Language α) : l∗ * l = l * l∗ := by
  /-
    α : Type u_1
    l : Language α
    ⊢ Eq (HMul.hMul (KStar.kstar l) l) (HMul.hMul l (KStar.kstar l))
  -/
  simp only [kstar_eq_iSup_pow, mul_iSup, iSup_mul, ← pow_succ, ← pow_succ']
  /-
    🎉 no goals
  -/


@[simp]
theorem one_add_self_mul_kstar_eq_kstar (l : Language α) : 1 + l * l∗ = l∗ := by
  /-
    α : Type u_1
    l : Language α
    ⊢ Eq (HAdd.hAdd 1 (HMul.hMul l (KStar.kstar l))) (KStar.kstar l)
  -/
  simp only [kstar_eq_iSup_pow, mul_iSup, ← pow_succ', ← pow_zero l]
  /-
    α : Type u_1
    l : Language α
    ⊢ Eq (HAdd.hAdd (HPow.hPow l 0) (iSup fun i => HPow.hPow l (HAdd.hAdd i 1))) ( …
  -/
  exact sup_iSup_nat_succ _
  /-
    🎉 no goals
  -/


@[simp]
theorem one_add_kstar_mul_self_eq_kstar (l : Language α) : 1 + l∗ * l = l∗ := by
  /-
    α : Type u_1
    l : Language α
    ⊢ Eq (HAdd.hAdd 1 (HMul.hMul (KStar.kstar l) l)) (KStar.kstar l)
  -/
  rw [mul_self_kstar_comm, one_add_self_mul_kstar_eq_kstar]
  /-
    🎉 no goals
  -/


instance : KleeneAlgebra (Language α) :=
  { instSemiring, instCompleteAtomicBooleanAlgebra with
    kstar := fun L ↦ L∗,
                                             /-
                                               α : Type u_1
                                               β : Type u_2
                                               γ : Type u_3
                                               l m : Language α
                                               a✝ b x : List α
                                               a : Language α
                                               x✝ : List α
                                               hl : Membership.mem 1 x✝
                                               ⊢ ∀ (y : List α), Membership.mem List.nil y → Membership.mem a y
                                             -/
    one_le_kstar := fun a _ hl ↦ ⟨[], hl, by simp⟩,
                                             /-
                                               🎉 no goals
                                             -/
    mul_kstar_le_kstar := fun a ↦ (one_add_self_mul_kstar_eq_kstar a).le.trans' le_sup_right,
    kstar_mul_le_kstar := fun a ↦ (one_add_kstar_mul_self_eq_kstar a).le.trans' le_sup_right,
    kstar_mul_le_self := fun l m h ↦ by
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        l✝ m✝ : Language α
        a b x : List α
        l m : Language α
        h : LE.le (HMul.hMul l m) m
        ⊢ LE.le (HMul.hMul (KStar.kstar l) m) m
      -/
      rw [kstar_eq_iSup_pow, iSup_mul]
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        l✝ m✝ : Language α
        a b x : List α
        l m : Language α
        h : LE.le (HMul.hMul l m) m
        ⊢ LE.le (iSup fun i => HMul.hMul (HPow.hPow l i) m) m
      -/
      refine iSup_le (fun n ↦ ?_)
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        l✝ m✝ : Language α
        a b x : List α
        l m : Language α
        h : LE.le (HMul.hMul l m) m
        n : Nat
        ⊢ LE.le (HMul.hMul (HPow.hPow l n) m) m
      -/
      induction' n with n ih
        /-
          case zero
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          l✝ m✝ : Language α
          a b x : List α
          l m : Language α
          h : LE.le (HMul.hMul l m) m
          ⊢ LE.le (HMul.hMul (HPow.hPow l 0) m) m
        -/
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        l✝ m✝ : Language α
        a b x : List α
        l m : Language α
        h : LE.le (HMul.hMul m l) m
        ⊢ LE.le (HMul.hMul m (KStar.kstar l)) m
      -/
      · simp
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        l✝ m✝ : Language α
        a b x : List α
        l m : Language α
        h : LE.le (HMul.hMul m l) m
        ⊢ LE.le (iSup fun i => HMul.hMul m (HPow.hPow l i)) m
      -/
        /-
          🎉 no goals
        -/
      /-
        case succ
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        l✝ m✝ : Language α
        a b x : List α
        l m : Language α
        h : LE.le (HMul.hMul l m) m
        n : Nat
        ih : LE.le (HMul.hMul (HPow.hPow l n) m) m
        ⊢ LE.le (HMul.hMul (HPow.hPow l (HAdd.hAdd n 1)) m) m
      -/
      rw [pow_succ, mul_assoc (l^n) l m]
      /-
        case succ
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        l✝ m✝ : Language α
        a b x : List α
        l m : Language α
        h : LE.le (HMul.hMul l m) m
        n : Nat
        ih : LE.le (HMul.hMul (HPow.hPow l n) m) m
        ⊢ LE.le (HMul.hMul (HPow.hPow l n) (HMul.hMul l m)) m
      -/
      exact le_trans (le_mul_congr le_rfl h) ih,
      /-
        🎉 no goals
      -/
    mul_kstar_le_self := fun l m h ↦ by
      rw [kstar_eq_iSup_pow, mul_iSup]
      refine iSup_le (fun n ↦ ?_)
      induction n with
      | zero => simp
      | succ n ih =>
        rw [pow_succ, ← mul_assoc m (l^n) l]
        exact le_trans (le_mul_congr ih le_rfl) h }


/-- Language `l.reverse` is defined as the set of words from `l` backwards. -/
def reverse (l : Language α) : Language α := { w : List α | w.reverse ∈ l }


@[simp]
lemma mem_reverse : a ∈ l.reverse ↔ a.reverse ∈ l := Iff.rfl


lemma reverse_mem_reverse : a.reverse ∈ l.reverse ↔ a ∈ l := by
  /-
    α : Type u_1
    l : Language α
    a : List α
    ⊢ Iff (Membership.mem l.reverse a.reverse) (Membership.mem l a)
  -/
  rw [mem_reverse, List.reverse_reverse]
  /-
    🎉 no goals
  -/


lemma reverse_eq_image (l : Language α) : l.reverse = List.reverse '' l :=
  ((List.reverse_involutive.toPerm _).image_eq_preimage _).symm


@[simp]
lemma reverse_zero : (0 : Language α).reverse = 0 := rfl


@[simp]
lemma reverse_one : (1 : Language α).reverse = 1 := by
  /-
    α : Type u_1
    ⊢ Eq (Language.reverse 1) 1
  -/
  simp [reverse, ← one_def]
  /-
    🎉 no goals
  -/


lemma reverse_involutive : Function.Involutive (reverse : Language α → _) :=
  List.reverse_involutive.preimage


lemma reverse_bijective : Function.Bijective (reverse : Language α → _) :=
  reverse_involutive.bijective


lemma reverse_injective : Function.Injective (reverse : Language α → _) :=
  reverse_involutive.injective


lemma reverse_surjective : Function.Surjective (reverse : Language α → _) :=
  reverse_involutive.surjective


@[simp]
lemma reverse_reverse (l : Language α) : l.reverse.reverse = l := reverse_involutive l


@[simp]
lemma reverse_add (l m : Language α) : (l + m).reverse = l.reverse + m.reverse := rfl


@[simp]
lemma reverse_mul (l m : Language α) : (l * m).reverse = m.reverse * l.reverse := by
  simp only [mul_def, reverse_eq_image, image2_image_left, image2_image_right, image_image2,
    List.reverse_append]
  /-
    α : Type u_1
    l m : Language α
    ⊢ Eq (Set.image2 (fun a b => HAppend.hAppend b.reverse a.reverse) l m) (Set.im …
  -/
  apply image2_swap
  /-
    🎉 no goals
  -/


@[simp]
lemma reverse_iSup {ι : Sort*} (l : ι → Language α) : (⨆ i, l i).reverse = ⨆ i, (l i).reverse :=
  preimage_iUnion


@[simp]
lemma reverse_iInf {ι : Sort*} (l : ι → Language α) : (⨅ i, l i).reverse = ⨅ i, (l i).reverse :=
  preimage_iInter


variable (α) in
/-- `Language.reverse` as a ring isomorphism to the opposite ring. -/
@[simps]
def reverseIso : Language α ≃+* (Language α)ᵐᵒᵖ where
  toFun l := .op l.reverse
  invFun l' := l'.unop.reverse
  left_inv := reverse_reverse
  right_inv l' := MulOpposite.unop_injective <| reverse_reverse l'.unop
  map_mul' l₁ l₂ := MulOpposite.unop_injective <| reverse_mul l₁ l₂
  map_add' l₁ l₂ := MulOpposite.unop_injective <| reverse_add l₁ l₂


@[simp]
lemma reverse_pow (l : Language α) (n : ℕ) : (l ^ n).reverse = l.reverse ^ n :=
  MulOpposite.op_injective (map_pow (reverseIso α) l n)


@[simp]
lemma reverse_kstar (l : Language α) : l∗.reverse = l.reverse∗ := by
  /-
    α : Type u_1
    l : Language α
    ⊢ Eq (KStar.kstar l).reverse (KStar.kstar l.reverse)
  -/
  simp only [kstar_eq_iSup_pow, reverse_iSup, reverse_pow]
  /-
    🎉 no goals
  -/


/-- Symbols for use by all kinds of grammars. -/
inductive Symbol (T N : Type*)
  /-- Terminal symbols (of the same type as the language) -/
  | terminal    (t : T) : Symbol T N
  /-- Nonterminal symbols (must not be present at the end of word being generated) -/
  | nonterminal (n : N) : Symbol T N
deriving
  DecidableEq, Repr, Fintype


