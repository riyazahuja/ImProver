instance Pi.noZeroSMulDivisors (α) [Zero α] [∀ i, Zero <| f i]
    [∀ i, SMulWithZero α <| f i] [∀ i, NoZeroSMulDivisors α <| f i] :
    NoZeroSMulDivisors α (∀ i : I, f i) :=
  ⟨fun {_ _} h =>
    or_iff_not_imp_left.mpr fun hc =>
      funext fun i => (smul_eq_zero.mp (congr_fun h i)).resolve_left hc⟩


/-- A special case of `Pi.noZeroSMulDivisors` for non-dependent types. Lean struggles to
synthesize this instance by itself elsewhere in the library. -/
instance _root_.Function.noZeroSMulDivisors {ι α β : Type*} [Zero α] [Zero β]
    [SMulWithZero α β] [NoZeroSMulDivisors α β] : NoZeroSMulDivisors α (ι → β) :=
  Pi.noZeroSMulDivisors _

