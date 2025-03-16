/-- An algebraically closed field `F` satisfies `HasEnoughRootsOfUnity F n` for all `n`
that are not divisible by the characteristic of `F`. -/
instance hasEnoughRootsOfUnity (F : Type*) [Field F] [IsAlgClosed F] (n : ℕ) [i : NeZero (n : F)] :
    HasEnoughRootsOfUnity F n where
  prim := by
    /-
      F : Type u_1
      inst✝¹ : Field F
      inst✝ : IsAlgClosed F
      n : Nat
      i : NeZero ↑n
      ⊢ Exists fun m => IsPrimitiveRoot m n
    -/
    have : NeZero n := .of_neZero_natCast F
    /-
      F : Type u_1
      inst✝¹ : Field F
      inst✝ : IsAlgClosed F
      n : Nat
      i : NeZero ↑n
      this : NeZero n
      ⊢ Exists fun m => IsPrimitiveRoot m n
    -/
    have := isCyclotomicExtension {⟨n, NeZero.pos n⟩} F fun _ h ↦ Set.mem_singleton_iff.mp h ▸ i
    /-
      F : Type u_1
      inst✝¹ : Field F
      inst✝ : IsAlgClosed F
      n : Nat
      i : NeZero ↑n
      this✝ : NeZero n
      this : IsCyclotomicExtension (Singleton.singleton ⟨n, ⋯⟩) F F
      ⊢ Exists fun m => IsPrimitiveRoot m n
    -/
    exact IsCyclotomicExtension.exists_prim_root (S := {(⟨n, NeZero.pos n⟩ : ℕ+)}) F rfl
    /-
      🎉 no goals
    -/
  cyc :=
    have : NeZero n := .of_neZero_natCast F
    rootsOfUnity.isCyclic F n


