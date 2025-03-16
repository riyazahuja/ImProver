local prefix:100 "s" => cs.simple

local prefix:100 "π" => cs.wordProd


private theorem exists_word_with_prod (w : W) : ∃ n ω, ω.length = n ∧ π ω = w := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    ⊢ Exists fun n => Exists fun ω => And (Eq ω.length n) (Eq (cs.wordProd ω) w)
  -/
  rcases cs.wordProd_surjective w with ⟨ω, rfl⟩
  /-
    case intro
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    ⊢ Exists fun n => Exists fun ω_1 => And (Eq ω_1.length n) (Eq (cs.wordProd ω_1 …
  -/
  use ω.length, ω
  /-
    🎉 no goals
  -/


/-- The length of `w`; i.e., the minimum number of simple reflections that
must be multiplied to form `w`. -/
noncomputable def length (w : W) : ℕ := Nat.find (cs.exists_word_with_prod w)


local prefix:100 "ℓ" => cs.length


theorem exists_reduced_word (w : W) : ∃ ω, ω.length = ℓ w ∧ w = π ω := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    ⊢ Exists fun ω => And (Eq ω.length (cs.length w)) (Eq w (cs.wordProd ω))
  -/
  have := Nat.find_spec (cs.exists_word_with_prod w)
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    this : Exists fun ω => And (Eq ω.length (Nat.find ⋯)) (Eq (cs.wordProd ω) w)
    ⊢ Exists fun ω => And (Eq ω.length (cs.length w)) (Eq w (cs.wordProd ω))
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem length_wordProd_le (ω : List B) : ℓ (π ω) ≤ ω.length :=
                                                        /-
                                                          B : Type u_1
                                                          W : Type u_2
                                                          inst✝ : Group W
                                                          M : CoxeterMatrix B
                                                          cs : CoxeterSystem M W
                                                          ω : List B
                                                          ⊢ And (Eq ω.length ω.length) (Eq (cs.wordProd ω) (cs.wordProd ω))
                                                        -/
  Nat.find_min' (cs.exists_word_with_prod (π ω)) ⟨ω, by tauto⟩
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp] theorem length_one : ℓ (1 : W) = 0 := Nat.eq_zero_of_le_zero (cs.length_wordProd_le [])


@[simp]
theorem length_eq_zero_iff {w : W} : ℓ w = 0 ↔ w = 1 := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    ⊢ Iff (Eq (cs.length w) 0) (Eq w 1)
  -/
  constructor
    /-
      case mp
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      ⊢ Eq (cs.length w) 0 → Eq w 1
    -/
  · intro h
    /-
      case mp
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      h : Eq (cs.length w) 0
      ⊢ Eq w 1
    -/
    rcases cs.exists_reduced_word w with ⟨ω, hω, rfl⟩
    /-
      case mp.intro.intro
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ω : List B
      h : Eq (cs.length (cs.wordProd ω)) 0
      hω : Eq ω.length (cs.length (cs.wordProd ω))
      ⊢ Eq (cs.wordProd ω) 1
    -/
    have : ω = [] := eq_nil_of_length_eq_zero (hω.trans h)
    /-
      case mp.intro.intro
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ω : List B
      h : Eq (cs.length (cs.wordProd ω)) 0
      hω : Eq ω.length (cs.length (cs.wordProd ω))
      this : Eq ω List.nil
      ⊢ Eq (cs.wordProd ω) 1
    -/
    rw [this, wordProd_nil]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      ⊢ Eq w 1 → Eq (cs.length w) 0
    -/
  · rintro rfl
    /-
      case mpr
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ⊢ Eq (cs.length 1) 0
    -/
    exact cs.length_one
    /-
      🎉 no goals
    -/


@[simp]
theorem length_inv (w : W) : ℓ (w⁻¹) = ℓ w := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    ⊢ Eq (cs.length (Inv.inv w)) (cs.length w)
  -/
  apply Nat.le_antisymm
    /-
      case h₁
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      ⊢ LE.le (cs.length (Inv.inv w)) (cs.length w)
    -/
  · rcases cs.exists_reduced_word w with ⟨ω, hω, rfl⟩
    /-
      case h₁.intro.intro
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ω : List B
      hω : Eq ω.length (cs.length (cs.wordProd ω))
      ⊢ LE.le (cs.length (Inv.inv (cs.wordProd ω))) (cs.length (cs.wordProd ω))
    -/
    have := cs.length_wordProd_le (List.reverse ω)
    /-
      case h₁.intro.intro
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ω : List B
      hω : Eq ω.length (cs.length (cs.wordProd ω))
      this : LE.le (cs.length (cs.wordProd ω.reverse)) ω.reverse.length
      ⊢ LE.le (cs.length (Inv.inv (cs.wordProd ω))) (cs.length (cs.wordProd ω))
    -/
    rwa [wordProd_reverse, length_reverse, hω] at this
    /-
      🎉 no goals
    -/
    /-
      case h₂
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      ⊢ LE.le (cs.length w) (cs.length (Inv.inv w))
    -/
  · rcases cs.exists_reduced_word w⁻¹ with ⟨ω, hω, h'ω⟩
    /-
      case h₂.intro.intro
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      ω : List B
      hω : Eq ω.length (cs.length (Inv.inv w))
      h'ω : Eq (Inv.inv w) (cs.wordProd ω)
      ⊢ LE.le (cs.length w) (cs.length (Inv.inv w))
    -/
    have := cs.length_wordProd_le (List.reverse ω)
    /-
      case h₂.intro.intro
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      ω : List B
      hω : Eq ω.length (cs.length (Inv.inv w))
      h'ω : Eq (Inv.inv w) (cs.wordProd ω)
      this : LE.le (cs.length (cs.wordProd ω.reverse)) ω.reverse.length
      ⊢ LE.le (cs.length w) (cs.length (Inv.inv w))
    -/
    rwa [wordProd_reverse, length_reverse, ← h'ω, hω, inv_inv] at this
    /-
      🎉 no goals
    -/


theorem length_mul_le (w₁ w₂ : W) :
    ℓ (w₁ * w₂) ≤ ℓ w₁ + ℓ w₂ := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w₁ w₂ : W
    ⊢ LE.le (cs.length (HMul.hMul w₁ w₂)) (HAdd.hAdd (cs.length w₁) (cs.length w₂))
  -/
  rcases cs.exists_reduced_word w₁ with ⟨ω₁, hω₁, rfl⟩
  /-
    case intro.intro
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w₂ : W
    ω₁ : List B
    hω₁ : Eq ω₁.length (cs.length (cs.wordProd ω₁))
    ⊢ LE.le (cs.length (HMul.hMul (cs.wordProd ω₁) w₂)) (HAdd.hAdd (cs.length (cs. …
  -/
  rcases cs.exists_reduced_word w₂ with ⟨ω₂, hω₂, rfl⟩
  /-
    case intro.intro.intro.intro
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω₁ : List B
    hω₁ : Eq ω₁.length (cs.length (cs.wordProd ω₁))
    ω₂ : List B
    hω₂ : Eq ω₂.length (cs.length (cs.wordProd ω₂))
    ⊢ LE.le (cs.length (HMul.hMul (cs.wordProd ω₁) (cs.wordProd ω₂))) (HAdd.hAdd ( …
  -/
  have := cs.length_wordProd_le (ω₁ ++ ω₂)
  /-
    case intro.intro.intro.intro
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω₁ : List B
    hω₁ : Eq ω₁.length (cs.length (cs.wordProd ω₁))
    ω₂ : List B
    hω₂ : Eq ω₂.length (cs.length (cs.wordProd ω₂))
    this : LE.le (cs.length (cs.wordProd (HAppend.hAppend ω₁ ω₂))) (HAppend.hAppen …
    ⊢ LE.le (cs.length (HMul.hMul (cs.wordProd ω₁) (cs.wordProd ω₂))) (HAdd.hAdd ( …
  -/
  simpa [hω₁, hω₂, wordProd_append] using this
  /-
    🎉 no goals
  -/


theorem length_mul_ge_length_sub_length (w₁ w₂ : W) :
    ℓ w₁ - ℓ w₂ ≤ ℓ (w₁ * w₂) := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w₁ w₂ : W
    ⊢ LE.le (HSub.hSub (cs.length w₁) (cs.length w₂)) (cs.length (HMul.hMul w₁ w₂))
  -/
  simpa [Nat.sub_le_of_le_add] using cs.length_mul_le (w₁ * w₂) w₂⁻¹
  /-
    🎉 no goals
  -/


theorem length_mul_ge_length_sub_length' (w₁ w₂ : W) :
    ℓ w₂ - ℓ w₁ ≤ ℓ (w₁ * w₂) := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w₁ w₂ : W
    ⊢ LE.le (HSub.hSub (cs.length w₂) (cs.length w₁)) (cs.length (HMul.hMul w₁ w₂))
  -/
  simpa [Nat.sub_le_of_le_add, add_comm] using cs.length_mul_le w₁⁻¹ (w₁ * w₂)
  /-
    🎉 no goals
  -/


theorem length_mul_ge_max (w₁ w₂ : W) :
    max (ℓ w₁ - ℓ w₂) (ℓ w₂ - ℓ w₁) ≤ ℓ (w₁ * w₂) :=
  max_le_iff.mpr ⟨length_mul_ge_length_sub_length _ _ _, length_mul_ge_length_sub_length' _ _ _⟩


/-- The homomorphism that sends each element `w : W` to the parity of the length of `w`.
(See `lengthParity_eq_ofAdd_length`.) -/
def lengthParity : W →* Multiplicative (ZMod 2) := cs.lift ⟨fun _ ↦ Multiplicative.ofAdd 1, by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ⊢ M.IsLiftable fun x => Multiplicative.ofAdd 1
  -/
  simp_rw [CoxeterMatrix.IsLiftable, ← ofAdd_add, (by decide : (1 + 1 : ZMod 2) = 0)]
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ⊢ ∀ (i i' : B), Eq (HPow.hPow (Multiplicative.ofAdd 0) (M.M i i')) 1
  -/
  simp⟩
  /-
    🎉 no goals
  -/


theorem lengthParity_simple (i : B) :
    cs.lengthParity (s i) = Multiplicative.ofAdd 1 := cs.lift_apply_simple _ _


theorem lengthParity_comp_simple :
    cs.lengthParity ∘ cs.simple = fun _ ↦ Multiplicative.ofAdd 1 := funext cs.lengthParity_simple


theorem lengthParity_eq_ofAdd_length (w : W) :
    cs.lengthParity w = Multiplicative.ofAdd (↑(ℓ w)) := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    ⊢ Eq (cs.lengthParity w) (Multiplicative.ofAdd ↑(cs.length w))
  -/
  rcases cs.exists_reduced_word w with ⟨ω, hω, rfl⟩
  rw [← hω, wordProd, map_list_prod, List.map_map, lengthParity_comp_simple, map_const',
    prod_replicate, ← ofAdd_nsmul, nsmul_one]


theorem length_mul_mod_two (w₁ w₂ : W) : ℓ (w₁ * w₂) % 2 = (ℓ w₁ + ℓ w₂) % 2 := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w₁ w₂ : W
    ⊢ Eq (HMod.hMod (cs.length (HMul.hMul w₁ w₂)) 2) (HMod.hMod (HAdd.hAdd (cs.len …
  -/
  rw [← ZMod.natCast_eq_natCast_iff', Nat.cast_add]
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w₁ w₂ : W
    ⊢ Eq (↑(cs.length (HMul.hMul w₁ w₂))) (HAdd.hAdd ↑(cs.length w₁) ↑(cs.length w …
  -/
  simpa only [lengthParity_eq_ofAdd_length, ofAdd_add] using map_mul cs.lengthParity w₁ w₂
  /-
    🎉 no goals
  -/


@[simp]
theorem length_simple (i : B) : ℓ (s i) = 1 := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    i : B
    ⊢ Eq (cs.length (cs.simple i)) 1
  -/
  apply Nat.le_antisymm
    /-
      case h₁
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i : B
      ⊢ LE.le (cs.length (cs.simple i)) 1
    -/
  · simpa using cs.length_wordProd_le [i]
    /-
      🎉 no goals
    -/
    /-
      case h₂
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i : B
      ⊢ LE.le 1 (cs.length (cs.simple i))
    -/
  · by_contra! length_lt_one
    have : cs.lengthParity (s i) = Multiplicative.ofAdd 0 := by
      rw [lengthParity_eq_ofAdd_length, Nat.lt_one_iff.mp length_lt_one, Nat.cast_zero]
    have : Multiplicative.ofAdd (0 : ZMod 2) = Multiplicative.ofAdd 1 :=
      this.symm.trans (cs.lengthParity_simple i)
    /-
      case h₂
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i : B
      length_lt_one : LT.lt (cs.length (cs.simple i)) 1
      this✝ : Eq (cs.lengthParity (cs.simple i)) (Multiplicative.ofAdd 0)
      this : Eq (Multiplicative.ofAdd 0) (Multiplicative.ofAdd 1)
      ⊢ False
    -/
    contradiction
    /-
      🎉 no goals
    -/


theorem length_eq_one_iff {w : W} : ℓ w = 1 ↔ ∃ i : B, w = s i := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    ⊢ Iff (Eq (cs.length w) 1) (Exists fun i => Eq w (cs.simple i))
  -/
  constructor
    /-
      case mp
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      ⊢ Eq (cs.length w) 1 → Exists fun i => Eq w (cs.simple i)
    -/
  · intro h
    /-
      case mp
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      h : Eq (cs.length w) 1
      ⊢ Exists fun i => Eq w (cs.simple i)
    -/
    rcases cs.exists_reduced_word w with ⟨ω, hω, rfl⟩
    /-
      case mp.intro.intro
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ω : List B
      h : Eq (cs.length (cs.wordProd ω)) 1
      hω : Eq ω.length (cs.length (cs.wordProd ω))
      ⊢ Exists fun i => Eq (cs.wordProd ω) (cs.simple i)
    -/
    rcases List.length_eq_one.mp (hω.trans h) with ⟨i, rfl⟩
    /-
      case mp.intro.intro.intro
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i : B
      h : Eq (cs.length (cs.wordProd (List.cons i List.nil))) 1
      hω : Eq (List.cons i List.nil).length (cs.length (cs.wordProd (List.cons i Lis …
      ⊢ Exists fun i_1 => Eq (cs.wordProd (List.cons i List.nil)) (cs.simple i_1)
    -/
    exact ⟨i, cs.wordProd_singleton i⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      ⊢ (Exists fun i => Eq w (cs.simple i)) → Eq (cs.length w) 1
    -/
  · rintro ⟨i, rfl⟩
    /-
      case mpr.intro
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i : B
      ⊢ Eq (cs.length (cs.simple i)) 1
    -/
    exact cs.length_simple i
    /-
      🎉 no goals
    -/


theorem length_mul_simple_ne (w : W) (i : B) : ℓ (w * s i) ≠ ℓ w := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    ⊢ Ne (cs.length (HMul.hMul w (cs.simple i))) (cs.length w)
  -/
  intro eq
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    eq : Eq (cs.length (HMul.hMul w (cs.simple i))) (cs.length w)
    ⊢ False
  -/
  have length_mod_two := cs.length_mul_mod_two w (s i)
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    eq : Eq (cs.length (HMul.hMul w (cs.simple i))) (cs.length w)
    length_mod_two : Eq (HMod.hMod (cs.length (HMul.hMul w (cs.simple i))) 2) (HMo …
    ⊢ False
  -/
  rw [eq, length_simple] at length_mod_two
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    eq : Eq (cs.length (HMul.hMul w (cs.simple i))) (cs.length w)
    length_mod_two : Eq (HMod.hMod (cs.length w) 2) (HMod.hMod (HAdd.hAdd (cs.leng …
    ⊢ False
  -/
  rcases Nat.mod_two_eq_zero_or_one (ℓ w) with even | odd
    /-
      case inl
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      i : B
      eq : Eq (cs.length (HMul.hMul w (cs.simple i))) (cs.length w)
      length_mod_two : Eq (HMod.hMod (cs.length w) 2) (HMod.hMod (HAdd.hAdd (cs.leng …
      even : Eq (HMod.hMod (cs.length w) 2) 0
      ⊢ False
    -/
  · rw [even, Nat.succ_mod_two_eq_one_iff.mpr even] at length_mod_two
    /-
      case inl
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      i : B
      eq : Eq (cs.length (HMul.hMul w (cs.simple i))) (cs.length w)
      length_mod_two : Eq 0 1
      even : Eq (HMod.hMod (cs.length w) 2) 0
      ⊢ False
    -/
    contradiction
    /-
      🎉 no goals
    -/
    /-
      case inr
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      i : B
      eq : Eq (cs.length (HMul.hMul w (cs.simple i))) (cs.length w)
      length_mod_two : Eq (HMod.hMod (cs.length w) 2) (HMod.hMod (HAdd.hAdd (cs.leng …
      odd : Eq (HMod.hMod (cs.length w) 2) 1
      ⊢ False
    -/
  · rw [odd, Nat.succ_mod_two_eq_zero_iff.mpr odd] at length_mod_two
    /-
      case inr
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      i : B
      eq : Eq (cs.length (HMul.hMul w (cs.simple i))) (cs.length w)
      length_mod_two : Eq 1 0
      odd : Eq (HMod.hMod (cs.length w) 2) 1
      ⊢ False
    -/
    contradiction
    /-
      🎉 no goals
    -/


theorem length_simple_mul_ne (w : W) (i : B) : ℓ (s i * w) ≠ ℓ w := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    ⊢ Ne (cs.length (HMul.hMul (cs.simple i) w)) (cs.length w)
  -/
  convert cs.length_mul_simple_ne w⁻¹ i using 1
    /-
      case h.e'_2
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      i : B
      ⊢ Eq (cs.length (HMul.hMul (cs.simple i) w)) (cs.length (HMul.hMul (Inv.inv w) …
    -/
  · convert cs.length_inv ?_ using 2
    /-
      case h.e'_2.h.e'_6
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      i : B
      ⊢ Eq (HMul.hMul (cs.simple i) w) (Inv.inv (HMul.hMul (Inv.inv w) (cs.simple i)))
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      i : B
      ⊢ Eq (cs.length w) (cs.length (Inv.inv w))
    -/
  · simp
    /-
      🎉 no goals
    -/


theorem length_mul_simple (w : W) (i : B) :
    ℓ (w * s i) = ℓ w + 1 ∨ ℓ (w * s i) + 1 = ℓ w := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    ⊢ Or (Eq (cs.length (HMul.hMul w (cs.simple i))) (HAdd.hAdd (cs.length w) 1))  …
  -/
  rcases Nat.lt_or_gt_of_ne (cs.length_mul_simple_ne w i) with lt | gt
  · -- lt : ℓ (w * s i) < ℓ w
    /-
      case inl
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      i : B
      lt : LT.lt (cs.length (HMul.hMul w (cs.simple i))) (cs.length w)
      ⊢ Or (Eq (cs.length (HMul.hMul w (cs.simple i))) (HAdd.hAdd (cs.length w) 1))  …
    -/
    right
    /-
      case inl.h
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      i : B
      lt : LT.lt (cs.length (HMul.hMul w (cs.simple i))) (cs.length w)
      ⊢ Eq (HAdd.hAdd (cs.length (HMul.hMul w (cs.simple i))) 1) (cs.length w)
    -/
    have length_ge := cs.length_mul_ge_length_sub_length w (s i)
    /-
      case inl.h
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      i : B
      lt : LT.lt (cs.length (HMul.hMul w (cs.simple i))) (cs.length w)
      length_ge : LE.le (HSub.hSub (cs.length w) (cs.length (cs.simple i))) (cs.leng …
      ⊢ Eq (HAdd.hAdd (cs.length (HMul.hMul w (cs.simple i))) 1) (cs.length w)
    -/
    simp only [length_simple, tsub_le_iff_right] at length_ge
    -- length_ge : ℓ w ≤ ℓ (w * s i) + 1
    /-
      case inl.h
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      i : B
      lt : LT.lt (cs.length (HMul.hMul w (cs.simple i))) (cs.length w)
      length_ge : LE.le (cs.length w) (HAdd.hAdd (cs.length (HMul.hMul w (cs.simple  …
      ⊢ Eq (HAdd.hAdd (cs.length (HMul.hMul w (cs.simple i))) 1) (cs.length w)
    -/
    omega
    /-
      🎉 no goals
    -/
  · -- gt : ℓ w < ℓ (w * s i)
    /-
      case inr
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      i : B
      gt : GT.gt (cs.length (HMul.hMul w (cs.simple i))) (cs.length w)
      ⊢ Or (Eq (cs.length (HMul.hMul w (cs.simple i))) (HAdd.hAdd (cs.length w) 1))  …
    -/
    left
    /-
      case inr.h
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      i : B
      gt : GT.gt (cs.length (HMul.hMul w (cs.simple i))) (cs.length w)
      ⊢ Eq (cs.length (HMul.hMul w (cs.simple i))) (HAdd.hAdd (cs.length w) 1)
    -/
    have length_le := cs.length_mul_le w (s i)
    /-
      case inr.h
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      i : B
      gt : GT.gt (cs.length (HMul.hMul w (cs.simple i))) (cs.length w)
      length_le : LE.le (cs.length (HMul.hMul w (cs.simple i))) (HAdd.hAdd (cs.lengt …
      ⊢ Eq (cs.length (HMul.hMul w (cs.simple i))) (HAdd.hAdd (cs.length w) 1)
    -/
    simp only [length_simple] at length_le
    -- length_le : ℓ (w * s i) ≤ ℓ w + 1
    /-
      case inr.h
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      i : B
      gt : GT.gt (cs.length (HMul.hMul w (cs.simple i))) (cs.length w)
      length_le : LE.le (cs.length (HMul.hMul w (cs.simple i))) (HAdd.hAdd (cs.lengt …
      ⊢ Eq (cs.length (HMul.hMul w (cs.simple i))) (HAdd.hAdd (cs.length w) 1)
    -/
    omega
    /-
      🎉 no goals
    -/


theorem length_simple_mul (w : W) (i : B) :
    ℓ (s i * w) = ℓ w + 1 ∨ ℓ (s i * w) + 1 = ℓ w := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    ⊢ Or (Eq (cs.length (HMul.hMul (cs.simple i) w)) (HAdd.hAdd (cs.length w) 1))  …
  -/
  have := cs.length_mul_simple w⁻¹ i
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    this : Or (Eq (cs.length (HMul.hMul (Inv.inv w) (cs.simple i))) (HAdd.hAdd (cs …
    ⊢ Or (Eq (cs.length (HMul.hMul (cs.simple i) w)) (HAdd.hAdd (cs.length w) 1))  …
  -/
  rwa [(by simp : w⁻¹ * (s i) = ((s i) * w)⁻¹), length_inv, length_inv] at this
  /-
    🎉 no goals
  -/


/-- The proposition that `ω` is reduced; that is, it has minimal length among all words that
represent the same element of `W`. -/
def IsReduced (ω : List B) : Prop := ℓ (π ω) = ω.length


@[simp]
theorem isReduced_reverse_iff (ω : List B) : cs.IsReduced (ω.reverse) ↔ cs.IsReduced ω := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    ⊢ Iff (cs.IsReduced ω.reverse) (cs.IsReduced ω)
  -/
  simp [IsReduced]
  /-
    🎉 no goals
  -/


theorem IsReduced.reverse {cs : CoxeterSystem M W} {ω : List B}
    (hω : cs.IsReduced ω) : cs.IsReduced (ω.reverse) :=
  (cs.isReduced_reverse_iff ω).mpr hω


theorem exists_reduced_word' (w : W) : ∃ ω : List B, cs.IsReduced ω ∧ w = π ω := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    ⊢ Exists fun ω => And (cs.IsReduced ω) (Eq w (cs.wordProd ω))
  -/
  rcases cs.exists_reduced_word w with ⟨ω, hω, rfl⟩
  /-
    case intro.intro
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    hω : Eq ω.length (cs.length (cs.wordProd ω))
    ⊢ Exists fun ω_1 => And (cs.IsReduced ω_1) (Eq (cs.wordProd ω) (cs.wordProd ω_ …
  -/
  use ω
  /-
    case h
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    hω : Eq ω.length (cs.length (cs.wordProd ω))
    ⊢ And (cs.IsReduced ω) (Eq (cs.wordProd ω) (cs.wordProd ω))
  -/
  tauto
  /-
    🎉 no goals
  -/


private theorem isReduced_take_and_drop {ω : List B} (hω : cs.IsReduced ω) (j : ℕ) :
    cs.IsReduced (ω.take j) ∧ cs.IsReduced (ω.drop j) := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    hω : cs.IsReduced ω
    j : Nat
    ⊢ And (cs.IsReduced (List.take j ω)) (cs.IsReduced (List.drop j ω))
  -/
  have h₁ : ℓ (π (ω.take j)) ≤ (ω.take j).length    := cs.length_wordProd_le (ω.take j)
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    hω : cs.IsReduced ω
    j : Nat
    h₁ : LE.le (cs.length (cs.wordProd (List.take j ω))) (List.take j ω).length
    ⊢ And (cs.IsReduced (List.take j ω)) (cs.IsReduced (List.drop j ω))
  -/
  have h₂ : ℓ (π (ω.drop j)) ≤ (ω.drop j).length    := cs.length_wordProd_le (ω.drop j)
  have h₃ := calc
    (ω.take j).length + (ω.drop j).length
    _ = ω.length                             := by rw [← List.length_append, ω.take_append_drop j]
    _ = ℓ (π ω)                              := hω.symm
    _ = ℓ (π (ω.take j) * π (ω.drop j))      := by rw [← cs.wordProd_append, ω.take_append_drop j]
    _ ≤ ℓ (π (ω.take j)) + ℓ (π (ω.drop j))  := cs.length_mul_le _ _
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    hω : cs.IsReduced ω
    j : Nat
    h₁ : LE.le (cs.length (cs.wordProd (List.take j ω))) (List.take j ω).length
    h₂ : LE.le (cs.length (cs.wordProd (List.drop j ω))) (List.drop j ω).length
    h₃ : LE.le (HAdd.hAdd (List.take j ω).length (List.drop j ω).length) (HAdd.hAd …
    ⊢ And (cs.IsReduced (List.take j ω)) (cs.IsReduced (List.drop j ω))
  -/
  unfold IsReduced
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    hω : cs.IsReduced ω
    j : Nat
    h₁ : LE.le (cs.length (cs.wordProd (List.take j ω))) (List.take j ω).length
    h₂ : LE.le (cs.length (cs.wordProd (List.drop j ω))) (List.drop j ω).length
    h₃ : LE.le (HAdd.hAdd (List.take j ω).length (List.drop j ω).length) (HAdd.hAd …
    ⊢ And (Eq (cs.length (cs.wordProd (List.take j ω))) (List.take j ω).length) (E …
  -/
  omega
  /-
    🎉 no goals
  -/


theorem IsReduced.take {cs : CoxeterSystem M W} {ω : List B} (hω : cs.IsReduced ω) (j : ℕ) :
    cs.IsReduced (ω.take j) :=
  (isReduced_take_and_drop _ hω _).1


theorem IsReduced.drop {cs : CoxeterSystem M W} {ω : List B} (hω : cs.IsReduced ω) (j : ℕ) :
    cs.IsReduced (ω.drop j) :=
  (isReduced_take_and_drop _ hω _).2


theorem not_isReduced_alternatingWord (i i' : B) {m : ℕ} (hM : M i i' ≠ 0) (hm : m > M i i') :
    ¬cs.IsReduced (alternatingWord i i' m) := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    i i' : B
    m : Nat
    hM : Ne (M.M i i') 0
    hm : GT.gt m (M.M i i')
    ⊢ Not (cs.IsReduced (CoxeterSystem.alternatingWord i i' m))
  -/
  induction' hm with m _ ih
  · -- Base case; m = M i i' + 1
    suffices h : ℓ (π (alternatingWord i i' (M i i' + 1))) < M i i' + 1 by
      unfold IsReduced
      rw [Nat.succ_eq_add_one, length_alternatingWord]
      omega
    /-
      case refl
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i i' : B
      m : Nat
      hM : Ne (M.M i i') 0
      ⊢ LT.lt (cs.length (cs.wordProd (CoxeterSystem.alternatingWord i i' (HAdd.hAdd …
    -/
    have : M i i' + 1 ≤ M i i' * 2 := by linarith [Nat.one_le_iff_ne_zero.mpr hM]
    /-
      case refl
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i i' : B
      m : Nat
      hM : Ne (M.M i i') 0
      this : LE.le (HAdd.hAdd (M.M i i') 1) (HMul.hMul (M.M i i') 2)
      ⊢ LT.lt (cs.length (cs.wordProd (CoxeterSystem.alternatingWord i i' (HAdd.hAdd …
    -/
    rw [cs.prod_alternatingWord_eq_prod_alternatingWord_sub i i' _ this]
    /-
      case refl
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i i' : B
      m : Nat
      hM : Ne (M.M i i') 0
      this : LE.le (HAdd.hAdd (M.M i i') 1) (HMul.hMul (M.M i i') 2)
      ⊢ LT.lt (cs.length (cs.wordProd (CoxeterSystem.alternatingWord i' i (HSub.hSub …
    -/
    have : M i i' * 2 - (M i i' + 1) = M i i' - 1 := by omega
    /-
      case refl
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i i' : B
      m : Nat
      hM : Ne (M.M i i') 0
      this✝ : LE.le (HAdd.hAdd (M.M i i') 1) (HMul.hMul (M.M i i') 2)
      this : Eq (HSub.hSub (HMul.hMul (M.M i i') 2) (HAdd.hAdd (M.M i i') 1)) (HSub. …
      ⊢ LT.lt (cs.length (cs.wordProd (CoxeterSystem.alternatingWord i' i (HSub.hSub …
    -/
    rw [this]
    calc
      ℓ (π (alternatingWord i' i (M i i' - 1)))
      _ ≤ (alternatingWord i' i (M i i' - 1)).length  := cs.length_wordProd_le _
      _ = M i i' - 1                                  := length_alternatingWord _ _ _
      _ ≤ M i i'                                      := Nat.sub_le _ _
      _ < M i i' + 1                                  := Nat.lt_succ_self _
  · -- Inductive step
    /-
      case step
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i i' : B
      m✝ : Nat
      hM : Ne (M.M i i') 0
      m : Nat
      a✝ : (M.M i i').succ.le m
      ih : Not (cs.IsReduced (CoxeterSystem.alternatingWord i i' m))
      ⊢ Not (cs.IsReduced (CoxeterSystem.alternatingWord i i' m.succ))
    -/
    contrapose! ih
    /-
      case step
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i i' : B
      m✝ : Nat
      hM : Ne (M.M i i') 0
      m : Nat
      a✝ : (M.M i i').succ.le m
      ih : cs.IsReduced (CoxeterSystem.alternatingWord i i' m.succ)
      ⊢ cs.IsReduced (CoxeterSystem.alternatingWord i i' m)
    -/
    rw [alternatingWord_succ'] at ih
    /-
      case step
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i i' : B
      m✝ : Nat
      hM : Ne (M.M i i') 0
      m : Nat
      a✝ : (M.M i i').succ.le m
      ih : cs.IsReduced (List.cons (ite (Even m) i' i) (CoxeterSystem.alternatingWor …
      ⊢ cs.IsReduced (CoxeterSystem.alternatingWord i i' m)
    -/
    apply IsReduced.drop (j := 1) at ih
    /-
      case step
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i i' : B
      m✝ : Nat
      hM : Ne (M.M i i') 0
      m : Nat
      a✝ : (M.M i i').succ.le m
      ih : cs.IsReduced (List.drop 1 (List.cons (ite (Even m) i' i) (CoxeterSystem.a …
      ⊢ cs.IsReduced (CoxeterSystem.alternatingWord i i' m)
    -/
    simpa using ih
    /-
      🎉 no goals
    -/


/-- The proposition that `i` is a left descent of `w`; that is, $\ell(s_i w) < \ell(w)$. -/
def IsLeftDescent (w : W) (i : B) : Prop := ℓ (s i * w) < ℓ w


/-- The proposition that `i` is a right descent of `w`; that is, $\ell(w s_i) < \ell(w)$. -/
def IsRightDescent (w : W) (i : B) : Prop := ℓ (w * s i) < ℓ w


                                                                    /-
                                                                      B : Type u_1
                                                                      W : Type u_2
                                                                      inst✝ : Group W
                                                                      M : CoxeterMatrix B
                                                                      cs : CoxeterSystem M W
                                                                      i : B
                                                                      ⊢ Not (cs.IsLeftDescent 1 i)
                                                                    -/
theorem not_isLeftDescent_one (i : B) : ¬cs.IsLeftDescent 1 i := by simp [IsLeftDescent]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


                                                                      /-
                                                                        B : Type u_1
                                                                        W : Type u_2
                                                                        inst✝ : Group W
                                                                        M : CoxeterMatrix B
                                                                        cs : CoxeterSystem M W
                                                                        i : B
                                                                        ⊢ Not (cs.IsRightDescent 1 i)
                                                                      -/
theorem not_isRightDescent_one (i : B) : ¬cs.IsRightDescent 1 i := by simp [IsRightDescent]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem isLeftDescent_inv_iff {w : W} {i : B} :
    cs.IsLeftDescent w⁻¹ i ↔ cs.IsRightDescent w i := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    ⊢ Iff (cs.IsLeftDescent (Inv.inv w) i) (cs.IsRightDescent w i)
  -/
  unfold IsLeftDescent IsRightDescent
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    ⊢ Iff (LT.lt (cs.length (HMul.hMul (cs.simple i) (Inv.inv w))) (cs.length (Inv …
  -/
  nth_rw 1 [← length_inv]
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    ⊢ Iff (LT.lt (cs.length (Inv.inv (HMul.hMul (cs.simple i) (Inv.inv w)))) (cs.l …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem isRightDescent_inv_iff {w : W} {i : B} :
    cs.IsRightDescent w⁻¹ i ↔ cs.IsLeftDescent w i := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    ⊢ Iff (cs.IsRightDescent (Inv.inv w) i) (cs.IsLeftDescent w i)
  -/
  simpa using (cs.isLeftDescent_inv_iff (w := w⁻¹)).symm
  /-
    🎉 no goals
  -/


theorem exists_leftDescent_of_ne_one {w : W} (hw : w ≠ 1) : ∃ i : B, cs.IsLeftDescent w i := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    hw : Ne w 1
    ⊢ Exists fun i => cs.IsLeftDescent w i
  -/
  rcases cs.exists_reduced_word w with ⟨ω, h, rfl⟩
  /-
    case intro.intro
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    hw : Ne (cs.wordProd ω) 1
    h : Eq ω.length (cs.length (cs.wordProd ω))
    ⊢ Exists fun i => cs.IsLeftDescent (cs.wordProd ω) i
  -/
  have h₁ : ω ≠ [] := by rintro rfl; simp at hw
  /-
    case intro.intro
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    hw : Ne (cs.wordProd ω) 1
    h : Eq ω.length (cs.length (cs.wordProd ω))
    h₁ : Ne ω List.nil
    ⊢ Exists fun i => cs.IsLeftDescent (cs.wordProd ω) i
  -/
  rcases List.exists_cons_of_ne_nil h₁ with ⟨i, ω', rfl⟩
  /-
    case intro.intro.intro.intro
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    i : B
    ω' : List B
    hw : Ne (cs.wordProd (List.cons i ω')) 1
    h : Eq (List.cons i ω').length (cs.length (cs.wordProd (List.cons i ω')))
    h₁ : Ne (List.cons i ω') List.nil
    ⊢ Exists fun i_1 => cs.IsLeftDescent (cs.wordProd (List.cons i ω')) i_1
  -/
  use i
  /-
    case h
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    i : B
    ω' : List B
    hw : Ne (cs.wordProd (List.cons i ω')) 1
    h : Eq (List.cons i ω').length (cs.length (cs.wordProd (List.cons i ω')))
    h₁ : Ne (List.cons i ω') List.nil
    ⊢ cs.IsLeftDescent (cs.wordProd (List.cons i ω')) i
  -/
  rw [IsLeftDescent, ← h, wordProd_cons, simple_mul_simple_cancel_left]
  calc
    ℓ (π ω') ≤ ω'.length                := cs.length_wordProd_le ω'
    _        < (i :: ω').length         := by simp


theorem exists_rightDescent_of_ne_one {w : W} (hw : w ≠ 1) : ∃ i : B, cs.IsRightDescent w i := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    hw : Ne w 1
    ⊢ Exists fun i => cs.IsRightDescent w i
  -/
  simp only [← isLeftDescent_inv_iff]
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    hw : Ne w 1
    ⊢ Exists fun i => cs.IsLeftDescent (Inv.inv w) i
  -/
  apply exists_leftDescent_of_ne_one
  /-
    case hw
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    hw : Ne w 1
    ⊢ Ne (Inv.inv w) 1
  -/
  simpa
  /-
    🎉 no goals
  -/


theorem isLeftDescent_iff {w : W} {i : B} :
    cs.IsLeftDescent w i ↔ ℓ (s i * w) + 1 = ℓ w := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    ⊢ Iff (cs.IsLeftDescent w i) (Eq (HAdd.hAdd (cs.length (HMul.hMul (cs.simple i …
  -/
  unfold IsLeftDescent
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    ⊢ Iff (LT.lt (cs.length (HMul.hMul (cs.simple i) w)) (cs.length w)) (Eq (HAdd. …
  -/
  constructor
    /-
      case mp
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      i : B
      ⊢ LT.lt (cs.length (HMul.hMul (cs.simple i) w)) (cs.length w) → Eq (HAdd.hAdd  …
    -/
  · intro _
    /-
      case mp
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      i : B
      a✝ : LT.lt (cs.length (HMul.hMul (cs.simple i) w)) (cs.length w)
      ⊢ Eq (HAdd.hAdd (cs.length (HMul.hMul (cs.simple i) w)) 1) (cs.length w)
    -/
    exact (cs.length_simple_mul w i).resolve_left (by omega)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      i : B
      ⊢ Eq (HAdd.hAdd (cs.length (HMul.hMul (cs.simple i) w)) 1) (cs.length w) → LT. …
    -/
  · omega
    /-
      🎉 no goals
    -/


theorem not_isLeftDescent_iff {w : W} {i : B} :
    ¬cs.IsLeftDescent w i ↔ ℓ (s i * w) = ℓ w + 1 := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    ⊢ Iff (Not (cs.IsLeftDescent w i)) (Eq (cs.length (HMul.hMul (cs.simple i) w)) …
  -/
  unfold IsLeftDescent
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    ⊢ Iff (Not (LT.lt (cs.length (HMul.hMul (cs.simple i) w)) (cs.length w))) (Eq  …
  -/
  constructor
    /-
      case mp
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      i : B
      ⊢ Not (LT.lt (cs.length (HMul.hMul (cs.simple i) w)) (cs.length w)) → Eq (cs.l …
    -/
  · intro _
    /-
      case mp
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      i : B
      a✝ : Not (LT.lt (cs.length (HMul.hMul (cs.simple i) w)) (cs.length w))
      ⊢ Eq (cs.length (HMul.hMul (cs.simple i) w)) (HAdd.hAdd (cs.length w) 1)
    -/
    exact (cs.length_simple_mul w i).resolve_right (by omega)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      i : B
      ⊢ Eq (cs.length (HMul.hMul (cs.simple i) w)) (HAdd.hAdd (cs.length w) 1) → Not …
    -/
  · omega
    /-
      🎉 no goals
    -/


theorem isRightDescent_iff {w : W} {i : B} :
    cs.IsRightDescent w i ↔ ℓ (w * s i) + 1 = ℓ w := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    ⊢ Iff (cs.IsRightDescent w i) (Eq (HAdd.hAdd (cs.length (HMul.hMul w (cs.simpl …
  -/
  unfold IsRightDescent
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    ⊢ Iff (LT.lt (cs.length (HMul.hMul w (cs.simple i))) (cs.length w)) (Eq (HAdd. …
  -/
  constructor
    /-
      case mp
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      i : B
      ⊢ LT.lt (cs.length (HMul.hMul w (cs.simple i))) (cs.length w) → Eq (HAdd.hAdd  …
    -/
  · intro _
    /-
      case mp
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      i : B
      a✝ : LT.lt (cs.length (HMul.hMul w (cs.simple i))) (cs.length w)
      ⊢ Eq (HAdd.hAdd (cs.length (HMul.hMul w (cs.simple i))) 1) (cs.length w)
    -/
    exact (cs.length_mul_simple w i).resolve_left (by omega)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      i : B
      ⊢ Eq (HAdd.hAdd (cs.length (HMul.hMul w (cs.simple i))) 1) (cs.length w) → LT. …
    -/
  · omega
    /-
      🎉 no goals
    -/


theorem not_isRightDescent_iff {w : W} {i : B} :
    ¬cs.IsRightDescent w i ↔ ℓ (w * s i) = ℓ w + 1 := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    ⊢ Iff (Not (cs.IsRightDescent w i)) (Eq (cs.length (HMul.hMul w (cs.simple i)) …
  -/
  unfold IsRightDescent
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    ⊢ Iff (Not (LT.lt (cs.length (HMul.hMul w (cs.simple i))) (cs.length w))) (Eq  …
  -/
  constructor
    /-
      case mp
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      i : B
      ⊢ Not (LT.lt (cs.length (HMul.hMul w (cs.simple i))) (cs.length w)) → Eq (cs.l …
    -/
  · intro _
    /-
      case mp
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      i : B
      a✝ : Not (LT.lt (cs.length (HMul.hMul w (cs.simple i))) (cs.length w))
      ⊢ Eq (cs.length (HMul.hMul w (cs.simple i))) (HAdd.hAdd (cs.length w) 1)
    -/
    exact (cs.length_mul_simple w i).resolve_right (by omega)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      B : Type u_1
      W : Type u_2
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      i : B
      ⊢ Eq (cs.length (HMul.hMul w (cs.simple i))) (HAdd.hAdd (cs.length w) 1) → Not …
    -/
  · omega
    /-
      🎉 no goals
    -/


theorem isLeftDescent_iff_not_isLeftDescent_mul {w : W} {i : B} :
    cs.IsLeftDescent w i ↔ ¬cs.IsLeftDescent (s i * w) i := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    ⊢ Iff (cs.IsLeftDescent w i) (Not (cs.IsLeftDescent (HMul.hMul (cs.simple i) w …
  -/
  rw [isLeftDescent_iff, not_isLeftDescent_iff, simple_mul_simple_cancel_left]
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    ⊢ Iff (Eq (HAdd.hAdd (cs.length (HMul.hMul (cs.simple i) w)) 1) (cs.length w)) …
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem isRightDescent_iff_not_isRightDescent_mul {w : W} {i : B} :
    cs.IsRightDescent w i ↔ ¬cs.IsRightDescent (w * s i) i := by
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    ⊢ Iff (cs.IsRightDescent w i) (Not (cs.IsRightDescent (HMul.hMul w (cs.simple  …
  -/
  rw [isRightDescent_iff, not_isRightDescent_iff, simple_mul_simple_cancel_right]
  /-
    B : Type u_1
    W : Type u_2
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    ⊢ Iff (Eq (HAdd.hAdd (cs.length (HMul.hMul w (cs.simple i))) 1) (cs.length w)) …
  -/
  tauto
  /-
    🎉 no goals
  -/


