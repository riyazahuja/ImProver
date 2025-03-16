/-- `updateFinset x s y` is the vector `x` with the coordinates in `s` changed to the values of `y`.
-/
def updateFinset (x : ∀ i, π i) (s : Finset ι) (y : ∀ i : ↥s, π i) (i : ι) : π i :=
  if hi : i ∈ s then y ⟨i, hi⟩ else x i


theorem updateFinset_def {s : Finset ι} {y} :
    updateFinset x s y = fun i ↦ if hi : i ∈ s then y ⟨i, hi⟩ else x i :=
  rfl


@[simp] theorem updateFinset_empty {y} : updateFinset x ∅ y = x :=
  rfl


theorem updateFinset_singleton {i y} :
    updateFinset x {i} y = Function.update x i (y ⟨i, mem_singleton_self i⟩) := by
  /-
    ι : Type u_1
    π : ι → Sort u_2
    x : (i : ι) → π i
    inst✝ : DecidableEq ι
    i : ι
    y : (i_1 : Subtype fun x => Membership.mem (Singleton.singleton i) x) → π ↑i_1
    ⊢ Eq (Function.updateFinset x (Singleton.singleton i) y) (Function.update x i  …
  -/
  congr with j
  /-
    case h
    ι : Type u_1
    π : ι → Sort u_2
    x : (i : ι) → π i
    inst✝ : DecidableEq ι
    i : ι
    y : (i_1 : Subtype fun x => Membership.mem (Singleton.singleton i) x) → π ↑i_1
    j : ι
    ⊢ Eq (Function.updateFinset x (Singleton.singleton i) y j) (Function.update x  …
  -/
  by_cases hj : j = i
    /-
      case pos
      ι : Type u_1
      π : ι → Sort u_2
      x : (i : ι) → π i
      inst✝ : DecidableEq ι
      i : ι
      y : (i_1 : Subtype fun x => Membership.mem (Singleton.singleton i) x) → π ↑i_1
      j : ι
      hj : Eq j i
      ⊢ Eq (Function.updateFinset x (Singleton.singleton i) y j) (Function.update x  …
    -/
  · cases hj
    /-
      case pos.refl
      ι : Type u_1
      π : ι → Sort u_2
      x : (i : ι) → π i
      inst✝ : DecidableEq ι
      i : ι
      y : (i_1 : Subtype fun x => Membership.mem (Singleton.singleton i) x) → π ↑i_1
      ⊢ Eq (Function.updateFinset x (Singleton.singleton i) y i) (Function.update x  …
    -/
    simp only [dif_pos, Finset.mem_singleton, update_self, updateFinset]
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      π : ι → Sort u_2
      x : (i : ι) → π i
      inst✝ : DecidableEq ι
      i : ι
      y : (i_1 : Subtype fun x => Membership.mem (Singleton.singleton i) x) → π ↑i_1
      j : ι
      hj : Not (Eq j i)
      ⊢ Eq (Function.updateFinset x (Singleton.singleton i) y j) (Function.update x  …
    -/
  · simp [hj, updateFinset]
    /-
      🎉 no goals
    -/


theorem update_eq_updateFinset {i y} :
    Function.update x i y = updateFinset x {i} (uniqueElim y) := by
  /-
    ι : Type u_1
    π : ι → Sort u_2
    x : (i : ι) → π i
    inst✝ : DecidableEq ι
    i : ι
    y : π i
    ⊢ Eq (Function.update x i y) (Function.updateFinset x (Singleton.singleton i)  …
  -/
  congr with j
  /-
    case h
    ι : Type u_1
    π : ι → Sort u_2
    x : (i : ι) → π i
    inst✝ : DecidableEq ι
    i : ι
    y : π i
    j : ι
    ⊢ Eq (Function.update x i y j) (Function.updateFinset x (Singleton.singleton i …
  -/
  by_cases hj : j = i
    /-
      case pos
      ι : Type u_1
      π : ι → Sort u_2
      x : (i : ι) → π i
      inst✝ : DecidableEq ι
      i : ι
      y : π i
      j : ι
      hj : Eq j i
      ⊢ Eq (Function.update x i y j) (Function.updateFinset x (Singleton.singleton i …
    -/
  · cases hj
    /-
      case pos.refl
      ι : Type u_1
      π : ι → Sort u_2
      x : (i : ι) → π i
      inst✝ : DecidableEq ι
      i : ι
      y : π i
      ⊢ Eq (Function.update x i y i) (Function.updateFinset x (Singleton.singleton i …
    -/
    simp only [dif_pos, Finset.mem_singleton, update_self, updateFinset]
    /-
      case pos.refl
      ι : Type u_1
      π : ι → Sort u_2
      x : (i : ι) → π i
      inst✝ : DecidableEq ι
      i : ι
      y : π i
      ⊢ Eq y (uniqueElim y ⟨i, ⋯⟩)
    -/
    exact uniqueElim_default (α := fun j : ({i} : Finset ι) => π j) y
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      π : ι → Sort u_2
      x : (i : ι) → π i
      inst✝ : DecidableEq ι
      i : ι
      y : π i
      j : ι
      hj : Not (Eq j i)
      ⊢ Eq (Function.update x i y j) (Function.updateFinset x (Singleton.singleton i …
    -/
  · simp [hj, updateFinset]
    /-
      🎉 no goals
    -/


theorem updateFinset_updateFinset {s t : Finset ι} (hst : Disjoint s t)
    {y : ∀ i : ↥s, π i} {z : ∀ i : ↥t, π i} :
    updateFinset (updateFinset x s y) t z =
    updateFinset x (s ∪ t) (Equiv.piFinsetUnion π hst ⟨y, z⟩) := by
  /-
    ι : Type u_1
    π : ι → Type u_2
    x : (i : ι) → π i
    inst✝ : DecidableEq ι
    s t : Finset ι
    hst : Disjoint s t
    y : (i : Subtype fun x => Membership.mem s x) → π ↑i
    z : (i : Subtype fun x => Membership.mem t x) → π ↑i
    ⊢ Eq (Function.updateFinset (Function.updateFinset x s y) t z) (Function.updat …
  -/
  set e := Equiv.Finset.union s t hst
  /-
    ι : Type u_1
    π : ι → Type u_2
    x : (i : ι) → π i
    inst✝ : DecidableEq ι
    s t : Finset ι
    hst : Disjoint s t
    y : (i : Subtype fun x => Membership.mem s x) → π ↑i
    z : (i : Subtype fun x => Membership.mem t x) → π ↑i
    e : Equiv (Sum (Subtype fun x => Membership.mem s x) (Subtype fun x => Members …
    ⊢ Eq (Function.updateFinset (Function.updateFinset x s y) t z) (Function.updat …
  -/
  congr with i
  /-
    case h
    ι : Type u_1
    π : ι → Type u_2
    x : (i : ι) → π i
    inst✝ : DecidableEq ι
    s t : Finset ι
    hst : Disjoint s t
    y : (i : Subtype fun x => Membership.mem s x) → π ↑i
    z : (i : Subtype fun x => Membership.mem t x) → π ↑i
    e : Equiv (Sum (Subtype fun x => Membership.mem s x) (Subtype fun x => Members …
    i : ι
    ⊢ Eq (Function.updateFinset (Function.updateFinset x s y) t z i) (Function.upd …
  -/
  by_cases his : i ∈ s <;> by_cases hit : i ∈ t <;>
    /-
      case pos
      ι : Type u_1
      π : ι → Type u_2
      x : (i : ι) → π i
      inst✝ : DecidableEq ι
      s t : Finset ι
      hst : Disjoint s t
      y : (i : Subtype fun x => Membership.mem s x) → π ↑i
      z : (i : Subtype fun x => Membership.mem t x) → π ↑i
      e : Equiv (Sum (Subtype fun x => Membership.mem s x) (Subtype fun x => Members …
      i : ι
      his : Membership.mem s i
      hit : Membership.mem t i
      ⊢ Eq (Function.updateFinset (Function.updateFinset x s y) t z i) (Function.upd …
    -/
    simp only [updateFinset, his, hit, dif_pos, dif_neg, Finset.mem_union, false_or, not_false_iff]
    /-
      🎉 no goals
    -/
    /-
      case pos
      ι : Type u_1
      π : ι → Type u_2
      x : (i : ι) → π i
      inst✝ : DecidableEq ι
      s t : Finset ι
      hst : Disjoint s t
      y : (i : Subtype fun x => Membership.mem s x) → π ↑i
      z : (i : Subtype fun x => Membership.mem t x) → π ↑i
      e : Equiv (Sum (Subtype fun x => Membership.mem s x) (Subtype fun x => Members …
      i : ι
      his : Membership.mem s i
      hit : Membership.mem t i
      ⊢ Eq (z ⟨i, ⋯⟩) (dite (Or True True) (fun h => (Equiv.piFinsetUnion π hst) { f …
    -/
  · exfalso; exact Finset.disjoint_left.mp hst his hit
             /-
               🎉 no goals
             -/
    /-
      case neg
      ι : Type u_1
      π : ι → Type u_2
      x : (i : ι) → π i
      inst✝ : DecidableEq ι
      s t : Finset ι
      hst : Disjoint s t
      y : (i : Subtype fun x => Membership.mem s x) → π ↑i
      z : (i : Subtype fun x => Membership.mem t x) → π ↑i
      e : Equiv (Sum (Subtype fun x => Membership.mem s x) (Subtype fun x => Members …
      i : ι
      his : Membership.mem s i
      hit : Not (Membership.mem t i)
      ⊢ Eq (y ⟨i, ⋯⟩) (dite (Or True False) (fun h => (Equiv.piFinsetUnion π hst) {  …
    -/
  · exact piCongrLeft_sum_inl (fun b : ↥(s ∪ t) => π b) e y z ⟨i, his⟩ |>.symm
    /-
      🎉 no goals
    -/
    /-
      case pos
      ι : Type u_1
      π : ι → Type u_2
      x : (i : ι) → π i
      inst✝ : DecidableEq ι
      s t : Finset ι
      hst : Disjoint s t
      y : (i : Subtype fun x => Membership.mem s x) → π ↑i
      z : (i : Subtype fun x => Membership.mem t x) → π ↑i
      e : Equiv (Sum (Subtype fun x => Membership.mem s x) (Subtype fun x => Members …
      i : ι
      his : Not (Membership.mem s i)
      hit : Membership.mem t i
      ⊢ Eq (z ⟨i, ⋯⟩) ((Equiv.piFinsetUnion π hst) { fst := y, snd := z } ⟨i, ⋯⟩)
    -/
  · exact piCongrLeft_sum_inr (fun b : ↥(s ∪ t) => π b) e y z ⟨i, hit⟩ |>.symm
    /-
      🎉 no goals
    -/


